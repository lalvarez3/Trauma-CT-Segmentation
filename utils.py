class InterpolateMode(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"


InterpolateModeSequence = Union[
    Sequence[Union[InterpolateMode, str]], InterpolateMode, str
]

class ResizedC(MapTransform, InvertibleTransform):

    backend = Resize.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.resizer = Resize(spatial_size=spatial_size, size_mode=size_mode)
        self.spatial_size = spatial_size

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode, align_corners in self.key_iterator(
            d, self.mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners
                    if align_corners is not None
                    else TraceKeys.NONE,
                },
            )
            init_slice = int(d[key].shape[-1]*0.15)
            end_slice = int(d[key].shape[-1]*0.1)
            # Reduce Size in Memory
            if key == "label":
                d[key] = d[key].astype(np.int8)
                if d[key].shape[-1] > 600: d[key] = d[key][:,:,:,init_slice:-end_slice] #

                if d["image_meta_dict"].get("PatientName", None) and d["image_meta_dict"]["PatientName"].startswith("NI") and len(d[key].shape) != 4:
                    # print(d[key].shape)
                    liver_channel = np.where((d[key] != 6), 0, d[key])
                    liver_channel = np.where((liver_channel == 6), 1, liver_channel)
                    # liver_channel = np.expand_dims(liver_channel, 0)
                    w, h, z = self.spatial_size
                    liver_channel = self.resizer(liver_channel, align_corners=align_corners)
                    background = np.ones((1, z, w, h), dtype=np.float16) - liver_channel
                    empty_injures = np.zeros((1, z, w, h), dtype=np.float16)
                    resized = [background, liver_channel, empty_injures]
                    d[key] = np.stack(resized).astype(np.int8).squeeze()

                else:
                    label = d[key]
                    w, h, z = self.spatial_size
                    resized = list()
                    background = np.ones((1, w, h, z), dtype=np.int8)
                    for i, channel in enumerate([0, 2]):  # TODO: desharcodead
                        resized.append(
                            self.resizer(
                                np.expand_dims(label[channel, :, :, :], 0),
                                align_corners=align_corners,
                            )
                        )

                    background -= resized[0] # + resized[1]
                    resized = [background] + resized
                    d[key] = np.stack(resized).astype(np.int8).squeeze()
            else:
                if d[key].shape[-1] > 600: d[key] = d[key][:,:,:,init_slice:-end_slice]
                d[key] = self.resizer(d[key], align_corners=align_corners)

        keys = ['spacing', 'original_affine', 'affine', 'spatial_shape', 'original_channel_dim', 'filename_or_obj']
        new_label_metadata = dict()
        for key in keys:
            new_label_metadata[key] = d["label_meta_dict"].get(key, 0)

        d["label_meta_dict"] = new_label_metadata

        if "PatientID" not in d["image_meta_dict"]:
            d["image_meta_dict"]["PatientID"] = "0"
        if "PatientName" not in d["image_meta_dict"]:
            d["image_meta_dict"]["PatientName"] = "0"
        if "SliceThickness" not in d["image_meta_dict"]:
            d["image_meta_dict"]["SliceThickness"] = "0"
        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[TraceKeys.ORIG_SIZE]
            mode = transform[TraceKeys.EXTRA_INFO]["mode"]
            align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
            # Create inverse transform
            inverse_transform = Resize(
                spatial_size=orig_size,
                mode=mode,
                align_corners=None
                if align_corners == TraceKeys.NONE
                else align_corners,
            )
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d



# %%
class adaptOverlay(MapTransform, InvertibleTransform):

    backend = Resize.backend

    def __init__(
        self,
        keys: KeysCollection,
        size_mode: str = "all",
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))

    def __adapt_overlay__(self, overlay_path, mha_path, label):
        import SimpleITK as sitk
        if label.shape[-1] == 6:
            return label
        # Load the mha
        mha_data = sitk.ReadImage(mha_path)
        mha_org = mha_data.GetOrigin()[-1]
        # Load the mha image
        mha_img = sitk.GetArrayFromImage(mha_data)
        original_z_size = mha_img.shape[0]

        # Load the overlay
        overlay_data = sitk.ReadImage(overlay_path)
        overlay_org = overlay_data.GetOrigin()[-1]

        overlay_init = np.abs(1/mha_data.GetSpacing()[-1]*(mha_org-overlay_org) )

        lower_bound = int(overlay_init)
        upper_bound = label.shape[-1]
        zeros_up = lower_bound
        zeros_down = original_z_size - (upper_bound + lower_bound)
        new = list()

        if zeros_up > 0:
            new.append(np.zeros((label.shape[0], label.shape[1], zeros_up), dtype=label.dtype))

        new.append(label)

        if zeros_down > 0:
            new.append(np.zeros((label.shape[0], label.shape[1], zeros_down), dtype=label.dtype))

        label = np.concatenate(new, axis=2)

        return label


    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mode, align_corners in self.key_iterator(
            d, self.mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners
                    if align_corners is not None
                    else TraceKeys.NONE,
                },
            )
            # Reduce Size in Memory
            if key == "label":
                d[key] = d[key].astype(np.int8)
                if d["image_meta_dict"].get("PatientName", None) and d["image_meta_dict"]["PatientName"].startswith("NI"):
                    file_path = d["label_meta_dict"]["filename_or_obj"]
                    data_path = d["image_meta_dict"]["filename_or_obj"]
                    d[key] = self.__adapt_overlay__(file_path, data_path, d[key])
        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[TraceKeys.ORIG_SIZE]
            mode = transform[TraceKeys.EXTRA_INFO]["mode"]
            align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
            # Create inverse transform
            inverse_transform = Resize(
                spatial_size=orig_size,
                mode=mode,
                align_corners=None
                if align_corners == TraceKeys.NONE
                else align_corners,
            )
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d
        
class KeepOnlyClass(MapTransform, InvertibleTransform):

    def __init__(
        self,
        keys: KeysCollection,
        class_to_keep: int,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.class_to_keep = class_to_keep

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = np.where(d[key] == 255, 1, 0)
            # values = d[key]
            # print("NP UNIQUE", np.unique(values))
            # print('BEFORE EYE', d[key].shape)
            # n_values = np.max(values) + 1
            # d[key]= np.eye(n_values)[values]
            # print('AFTER EYE', d[key].shape)
            # d[key] = np.squeeze(d[key])
            # print('AFTER squeeze', d[key].shape)
            # print("NP UNIQUE AFTER", np.unique(d[key]))
            # if d[key].ndim == 2:
            #     zeros = np.zeros(d[key].shape)
            #     d[key] = np.stack([d[key], zeros], axis=-1)
            #     print(d[key].shape)
            # print("NP UNIQUE AFTER AFTER 0", np.unique(d[key][:,:,0]))
            # print("NP UNIQUE AFTER AFTER 1", np.unique(d[key][:,:,1]))

        return d
