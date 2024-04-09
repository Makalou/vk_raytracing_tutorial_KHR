#version 450
#extension GL_GOOGLE_include_directive : enable

#define VULKAN 100

layout(set = 0, binding = 0) uniform _GlobalUniforms
{
  mat4 viewProj;     // Camera view * projection
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix
}uni;

struct PICParticle
{
  vec4 position;
  vec4 velocity;
};

layout(set = 1, binding = 0) buffer _ParticleArray
{
    PICParticle p[];
}particles;

layout(location = 0) out vec3 color;

void main()
{
    vec4 particlePos = vec4(particles.p[gl_VertexIndex].position.xyz,1.0);
    gl_Position = uni.viewProj * particlePos;
    gl_PointSize = 10.0;
    color = vec3(abs(particles.p[gl_VertexIndex].velocity.xyz));
}