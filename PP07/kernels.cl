__kernel void Copy(__global ushort *restrict DstLm,
                   __global ushort *restrict DstCb,
                   __global ushort *restrict DstCr,
                   __global const ushort *restrict SrcLm,
                   __global const ushort *restrict SrcCb,
                   __global const ushort *restrict SrcCr, const int Width,
                   const int Height, const int Margin, const int Stride) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= Width || y >= Height) {
    return;
  }
  const int Offset = ((Margin + y) * Stride) + Margin + x;

  DstLm[Offset] = SrcLm[Offset];
  DstCb[Offset] = SrcCb[Offset];
  DstCr[Offset] = SrcCr[Offset];
}

//===============================================================================================================================================================================================================

__kernel void YUVtoRGBtoYUV(__global ushort *restrict DstLm,
                            __global ushort *restrict DstCb,
                            __global ushort *restrict DstCr,
                            __global const ushort *restrict SrcLm,
                            __global const ushort *restrict SrcCb,
                            __global const ushort *restrict SrcCr,
                            const uint Width, const uint Height,
                            const uint Margin, const uint Stride) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= Width || y >= Height) {
    return;
  }
  const int Offset = ((Margin + y) * Stride) + Margin + x;

  // load
  const ushort4 SrcPixelYUV_U16 = {SrcLm[Offset], SrcCb[Offset], SrcCr[Offset],
                                   0};

  // change data format (and remove chroma offset)
  const float4 SrcPixelYUV_F32 =
      (convert_float4(SrcPixelYUV_U16) - (float4)(0, 128, 128, 0));

  // convert YCbCr --> RGB (BT.709)
  /*
  const flt32V4 SrcPixelRGB_F32 =
                    {
                            SrcPixelYUV_F32[0] + SrcPixelYUV_F32[2] *
  YCbCr2RGB_F32[0][2], SrcPixelYUV_F32[0] + SrcPixelYUV_F32[1] *
  YCbCr2RGB_F32[1][1] + SrcPixelYUV_F32[2] * YCbCr2RGB_F32[1][2],
                            SrcPixelYUV_F32[0] + SrcPixelYUV_F32[1] *
  YCbCr2RGB_F32[2][1],
                            0
                    };
  static constexpr inline flt32 YCbCr2RGB_F32[3][3] =  //BT709
  {
    {(flt32)(1.0), (flt32)( 0      ), (flt32)( 1.5748 )}, //R_Y, R_U, R_V
    {(flt32)(1.0), (flt32)(-0.18732), (flt32)(-0.46812)}, //G_Y, G_U, G_V
    {(flt32)(1.0), (flt32)( 1.8556 ), (flt32)( 0      )}, //B_Y, B_U, B_V
  };
  */

  const float4 SrcPixelRGB_F32 = {
      // SrcPixelYUV_F32.x + SrcPixelYUV_F32.z * YCbCr[0][2], -- this one
      // doesn't work, I'll take the values from xSepia.h
      SrcPixelYUV_F32.x + SrcPixelYUV_F32.z * 1.5748,
      SrcPixelYUV_F32.x + SrcPixelYUV_F32.y * (-0.18732) +
          SrcPixelRGB_F32.z * (-0.46812),
      SrcPixelYUV_F32.x + SrcPixelYUV_F32.y * 1.8556, 0};

  // copy RGB --> RGB
  const float4 DstPixelRGB_F32 = SrcPixelRGB_F32;

  // convert RGB --> YCbCr
  /*
  static constexpr inline flt32 RGB2YCbCr_F32[3][3] = //BT709
  {
    {(flt32)( 0.21260), (flt32)( 0.71520), (flt32)( 0.07220)}, //Y_R, Y_G, Y_B
    {(flt32)(-0.11457), (flt32)(-0.38543), (flt32)( 0.50000)}, //U_R, U_G ,U_B
    {(flt32)( 0.50000), (flt32)(-0.45415), (flt32)(-0.04585)}, //V_R, V_B, V_G
  };
  */
  const float4 DstPixelYUV_F32 = {
      DstPixelRGB_F32.x * 0.21260 + DstPixelRGB_F32.y * 0.71520 +
          DstPixelRGB_F32.z * 0.07220,
      DstPixelRGB_F32.x * (-0.11457) + DstPixelRGB_F32.y * (-0.38543) +
          DstPixelRGB_F32.z * 0.50000,
      DstPixelRGB_F32.x * 0.50000 + DstPixelRGB_F32.y * (-0.45415) +
          DstPixelRGB_F32.z * (-0.04585),
      0};

  // copy YUV --> YUV
  // const float4 DstPixelYUV_F32 = SrcPixelYUV_F32; REDEFINITION OF
  // DstPixelYUV_F32

  const ushort4 DstPixel_U16 =
      convert_ushort4_sat(DstPixelYUV_F32 + (float4)(0, 128, 128, 0));

  DstLm[Offset] = DstPixel_U16.x;
  DstCb[Offset] = DstPixel_U16.y;
  DstCr[Offset] = DstPixel_U16.z;
}

//===============================================================================================================================================================================================================

__kernel void Sepia(__global ushort *restrict DstLm,
                    __global ushort *restrict DstCb,
                    __global ushort *restrict DstCr,
                    __global const ushort *restrict SrcLm,
                    __global const ushort *restrict SrcCb,
                    __global const ushort *restrict SrcCr, const uint Width,
                    const uint Height, const uint Margin, const uint Stride) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= Width || y >= Height) {
    return;
  }
  const int Offset = ((Margin + y) * Stride) + Margin + x;

  // load
  const ushort4 SrcPixelYUV_U16 = {SrcLm[Offset], SrcCb[Offset], SrcCr[Offset],
                                   0};

  // change data format (and remove chroma offset)
  const float4 SrcPixelYUV_F32 =
      (convert_float4(SrcPixelYUV_U16) - (float4)(0, 128, 128, 0));

  // convert YCbCr --> RGB (BT.709)
  /*
  const flt32V4 SrcPixelRGB_F32 =
                    {
                            SrcPixelYUV_F32[0] + SrcPixelYUV_F32[2] *
  YCbCr2RGB_F32[0][2], SrcPixelYUV_F32[0] + SrcPixelYUV_F32[1] *
  YCbCr2RGB_F32[1][1] + SrcPixelYUV_F32[2] * YCbCr2RGB_F32[1][2],
                            SrcPixelYUV_F32[0] + SrcPixelYUV_F32[1] *
  YCbCr2RGB_F32[2][1],
                            0
                    };
  static constexpr inline flt32 YCbCr2RGB_F32[3][3] =  //BT709
  {
    {(flt32)(1.0), (flt32)( 0      ), (flt32)( 1.5748 )}, //R_Y, R_U, R_V
    {(flt32)(1.0), (flt32)(-0.18732), (flt32)(-0.46812)}, //G_Y, G_U, G_V
    {(flt32)(1.0), (flt32)( 1.8556 ), (flt32)( 0      )}, //B_Y, B_U, B_V
  };
  */

  const float4 SrcPixelRGB_F32 = {
      // SrcPixelYUV_F32.x + SrcPixelYUV_F32.z * YCbCr[0][2], -- this one
      // doesn't work, I'll take the values from xSepia.h
      SrcPixelYUV_F32.x + SrcPixelYUV_F32.z * 1.5748,
      SrcPixelYUV_F32.x + SrcPixelYUV_F32.y * (-0.18732) +
          SrcPixelRGB_F32.z * (-0.46812),
      SrcPixelYUV_F32.x + SrcPixelYUV_F32.y * 1.8556, 0};

  // SEPIA
  /*
  static constexpr inline flt32 SEPIA_F32[3][3] = //SEPIA
  {
    {(flt32)(0.393), (flt32)(0.769), (flt32)(0.189)},
    {(flt32)(0.349), (flt32)(0.686), (flt32)(0.168)},
    {(flt32)(0.272), (flt32)(0.534), (flt32)(0.131)},
  };
  */
  const float4 DstPixelRGB_F32 = {
      SrcPixelRGB_F32.x * 0.393 + SrcPixelRGB_F32.y * 0.769 +
          SrcPixelRGB_F32.z * 0.189,
      SrcPixelRGB_F32.x * 0.349 + SrcPixelRGB_F32.y * 0.686 +
          SrcPixelRGB_F32.z * 0.168,
      SrcPixelRGB_F32.x * 0.272 + SrcPixelRGB_F32.y * 0.534 +
          SrcPixelRGB_F32.z * 0.131,
      0};

  // convert RGB --> YCbCr
  /*
  static constexpr inline flt32 RGB2YCbCr_F32[3][3] = //BT709
  {
    {(flt32)( 0.21260), (flt32)( 0.71520), (flt32)( 0.07220)}, //Y_R, Y_G, Y_B
    {(flt32)(-0.11457), (flt32)(-0.38543), (flt32)( 0.50000)}, //U_R, U_G ,U_B
    {(flt32)( 0.50000), (flt32)(-0.45415), (flt32)(-0.04585)}, //V_R, V_B, V_G
  };
  */
  const float4 DstPixelYUV_F32 = {
      DstPixelRGB_F32.x * 0.21260 + DstPixelRGB_F32.y * 0.71520 +
          DstPixelRGB_F32.z * 0.07220,
      DstPixelRGB_F32.x * (-0.11457) + DstPixelRGB_F32.y * (-0.38543) +
          DstPixelRGB_F32.z * 0.50000,
      DstPixelRGB_F32.x * 0.50000 + DstPixelRGB_F32.y * (-0.45415) +
          DstPixelRGB_F32.z * (-0.04585),
      0};

  // copy YUV --> YUV
  // const float4 DstPixelYUV_F32 = SrcPixelYUV_F32; REDEFINITION OF
  // DstPixelYUV_F32

  const ushort4 DstPixel_U16 =
      convert_ushort4_sat(DstPixelYUV_F32 + (float4)(0, 128, 128, 0));

  DstLm[Offset] = DstPixel_U16.x;
  DstCb[Offset] = DstPixel_U16.y;
  DstCr[Offset] = DstPixel_U16.z;
}