class_head = []
reg_max = 16

for idx, feature in enumerate(features):
    # --- First ConvBNSiLU ---
    conv_w   = weights[f"model_24_cv3_{idx}_0_conv_weight"].astype(np.float32)
    conv_b   = None
    bn_gamma = weights[f"model_24_cv3_{idx}_0_bn_weight"].astype(np.float32)
    bn_beta  = weights[f"model_24_cv3_{idx}_0_bn_bias"].astype(np.float32)
    bn_mean  = weights[f"model_24_cv3_{idx}_0_bn_running_mean"].astype(np.float32)
    bn_var   = weights[f"model_24_cv3_{idx}_0_bn_running_var"].astype(np.float32)

    x = ConvBNSiLU(
        feature,
        conv_w=conv_w,
        conv_b=conv_b,
        bn_w=bn_gamma,
        bn_b=bn_beta,
        bn_rm=bn_mean,
        bn_rv=bn_var,
        stride=1,
        padding=0
    )

    # --- Second ConvBNSiLU ---
    conv_w   = weights[f"model_24_cv3_{idx}_1_conv_weight"].astype(np.float32)
    conv_b   = None
    bn_gamma = weights[f"model_24_cv3_{idx}_1_bn_weight"].astype(np.float32)
    bn_beta  = weights[f"model_24_cv3_{idx}_1_bn_bias"].astype(np.float32)
    bn_mean  = weights[f"model_24_cv3_{idx}_1_bn_running_mean"].astype(np.float32)
    bn_var   = weights[f"model_24_cv3_{idx}_1_bn_running_var"].astype(np.float32)

    y = ConvBNSiLU(
        x,
        conv_w=conv_w,
        conv_b=conv_b,
        bn_w=bn_gamma,
        bn_b=bn_beta,
        bn_rm=bn_mean,
        bn_rv=bn_var,
        stride=1,
        padding=0
    )

    # --- Final Conv 1x1 ---
    weight = weights[f"model_24_cv3_{idx}_2_weight"].astype(np.float32)
    bias   = weights[f"model_24_cv3_{idx}_2_bias"].astype(np.float32)
    z = Conv2d(y, weight, bias, stride=1, padding=0)
    
    class.append(z)
    
#  c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
#         )
#         self.cv3 = (
#             nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#    
# 1. Determine batch size
batch = class[0].shape[0]  # batch size

# 2. Reshape each box prediction
class_reshaped = []
for b in class:
    # b: shape [batch, channels, H, W]
    reshaped = b.reshape(batch, 4*reg_max, -1)  # flatten H*W into last dim
    class_reshaped.append(reshaped)

# 3. Concatenate predictions from all feature maps
boxes = np.concatenate(class_reshaped, axis=-1)  # shape: [batch, 4*reg_max, total_anchors]

# 4. Print shapes for verification
for idx, b in enumerate(class):
    print(f"Feature {idx} raw box pred shape: {b.shape}, reshaped: {class_reshaped[idx].shape}")
print("Final concatenated boxes shape:", boxes.shape)