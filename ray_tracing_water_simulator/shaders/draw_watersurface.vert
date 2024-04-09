#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "gltf.glsl"
#include "host_device.h"

#define VULKAN 100

layout(set = 0, binding = 0) uniform _GlobalUniforms
{
  GlobalUniforms uni;
};

layout(location = 0) in vec3 i_position;
layout(location = 1) in vec3 i_normal;

layout(location = 1) out vec3 o_worldPos;
layout(location = 2) out vec3 o_worldNrm;
layout(location = 3) out vec3 o_viewDir;

out gl_PerVertex
{
  vec4 gl_Position;
};

void main()
{
  vec3 origin = vec3(uni.viewInverse * vec4(0, 0, 0, 1));

  o_worldPos = i_position;
  o_viewDir  = vec3(o_worldPos - origin);
  o_worldNrm = i_normal;

  gl_Position = uni.viewProj * vec4(o_worldPos, 1.0);
}