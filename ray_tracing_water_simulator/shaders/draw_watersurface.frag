#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "gltf.glsl"
#include "host_device.h"

#define VULKAN 100

layout(push_constant) uniform _PushConstantRaster
{
  PushConstantRaster pcRaster;
};

// clang-format off
// Incoming 
layout(location = 1) in vec3 i_worldPos;
layout(location = 2) in vec3 i_worldNrm;
layout(location = 3) in vec3 i_viewDir;
// Outgoing
layout(location = 0) out vec4 o_color;
// clang-format on

void main()
{
  GltfShadeMaterial mat;
  mat.pbrBaseColorFactor = vec4(0,0,0.5,1);
  vec3 N = normalize(i_worldNrm);

  o_color = vec4(vec3(0,0,1),1);
  return;
  // Vector toward light
  vec3  L;
  float lightIntensity = pcRaster.lightIntensity;
  if(pcRaster.lightType == 0)
  {
    vec3  lDir     = pcRaster.lightPosition - i_worldPos;
    float d        = length(lDir);
    lightIntensity = pcRaster.lightIntensity / (d * d);
    L              = normalize(lDir);
  }
  else
  {
    L = normalize(pcRaster.lightPosition);
  }

  // Diffuse
  vec3 diffuse = computeDiffuse(mat, L, N);

  // Specular
  vec3 specular = computeSpecular(mat, i_viewDir, L, N);

  // Result
  o_color = vec4(lightIntensity * (diffuse + specular), 1);
}