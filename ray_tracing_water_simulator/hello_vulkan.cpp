/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#include <sstream>


#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/gltfscene.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "nvh/alignment.hpp"
#include "nvvk/buffers_vk.hpp"

#include <iostream>

extern std::vector<std::string> defaultSearchPaths;

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  glm::mat4      proj        = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspectRatio, 0.1f, 1000.0f);
  proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = glm::inverse(view);
  hostUBO.projInverse = glm::inverse(proj);

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  auto& bind = m_descSetLayoutBind;
  // Camera matrices
  bind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  // Array of textures
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTextures,
                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
  // Scene buffers
  bind.addBinding(eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                      | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo sceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, eSceneDesc, &sceneDesc));

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures)
    diit.emplace_back(texture.descriptor);
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);

  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescriptions({{0, sizeof(glm::vec3)}, {1, sizeof(glm::vec3)}, {2, sizeof(glm::vec2)}});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Position
      {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Normal
      {2, 2, VK_FORMAT_R32G32_SFLOAT, 0},     // Texcoord0
  });
  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");

  VkPipelineLayoutCreateInfo drawParticlePipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkDescriptorSetLayout      layouts[2]               = {m_descSetLayout, m_waterSimParticleDescriptorSetLayout};
  drawParticlePipelineLayoutCreateInfo.setLayoutCount = 2;
  drawParticlePipelineLayoutCreateInfo.pSetLayouts    = layouts;
  vkCreatePipelineLayout(m_device, &drawParticlePipelineLayoutCreateInfo, nullptr, &m_drawParticlePipelineLayout);

  nvvk::GraphicsPipelineGeneratorCombined gpb2(m_device, m_drawParticlePipelineLayout, m_offscreenRenderPass);
  gpb2.depthStencilState.depthTestEnable = true;
  gpb2.inputAssemblyState.topology       = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  gpb2.rasterizationState.polygonMode    = VK_POLYGON_MODE_POINT;
  gpb2.addShader(nvh::loadFile("spv/draw_particle.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb2.addShader(nvh::loadFile("spv/draw_particle.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  m_drawParticlePipeline = gpb2.createPipeline();
  m_debug.setObjectName(m_drawParticlePipeline, "DrawParticle");

  nvvk::GraphicsPipelineGeneratorCombined gpb3(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb3.depthStencilState.depthTestEnable = true;
  gpb3.inputAssemblyState.topology       = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  gpb3.rasterizationState.cullMode       = VK_CULL_MODE_NONE;
  gpb3.rasterizationState.polygonMode    = VK_POLYGON_MODE_FILL;
  gpb3.addShader(nvh::loadFile("spv/draw_watersurface.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb3.addShader(nvh::loadFile("spv/draw_watersurface.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb3.addBindingDescriptions({{0, sizeof(glm::vec3)}, {1, sizeof(glm::vec3)}});
  gpb3.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Position
      {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Normal
  });
  m_drawWaterSurfacePipeline = gpb3.createPipeline();
  m_debug.setObjectName(m_drawWaterSurfacePipeline, "DrawWaterSurface");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadScene(const std::string& filename)
{
  using vkBU = VkBufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  LOGI("Loading file: %s", filename.c_str());
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    assert(!"Error while loading scene");
  }
  LOGW("%s", warn.c_str());
  LOGE("%s", error.c_str());


  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_indexBuffer  = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // Copying all materials, only the elements we need
  std::vector<GltfShadeMaterial> shadeMaterials;
  for(const auto& m : m_gltfScene.m_materials)
  {
    shadeMaterials.emplace_back(GltfShadeMaterial{m.baseColorFactor, m.emissiveFactor, m.baseColorTexture});
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // The following is used to find the primitive mesh information in the CHIT
  std::vector<PrimMeshInfo> primLookup;
  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
  }
  m_primInfo = m_alloc.createBuffer(cmdBuf, primLookup, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);


  SceneDesc sceneDesc;
  sceneDesc.vertexAddress   = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
  sceneDesc.indexAddress    = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);
  sceneDesc.normalAddress   = nvvk::getBufferDeviceAddress(m_device, m_normalBuffer.buffer);
  sceneDesc.uvAddress       = nvvk::getBufferDeviceAddress(m_device, m_uvBuffer.buffer);
  sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
  sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
  sceneDesc.waterVertexAddress = nvvk::getBufferDeviceAddress(m_device, m_marchingCubeVertexBuffer.buffer);
  sceneDesc.waterIndexAddress = nvvk::getBufferDeviceAddress(m_device, m_marchingCubeIndexBuffer.buffer);
  sceneDesc.waterNormalAddress = nvvk::getBufferDeviceAddress(m_device, m_marchingCubeNormalBuffer.buffer);
  m_sceneDesc               = m_alloc.createBuffer(cmdBuf, sizeof(SceneDesc), &sceneDesc,
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // Creates all textures found
  createTextureImages(cmdBuf, tmodel);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();


  NAME_VK(m_vertexBuffer.buffer);
  NAME_VK(m_indexBuffer.buffer);
  NAME_VK(m_normalBuffer.buffer);
  NAME_VK(m_uvBuffer.buffer);
  NAME_VK(m_materialBuffer.buffer);
  NAME_VK(m_primInfo.buffer);
  NAME_VK(m_sceneDesc.buffer);
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

  auto addDefaultTexture = [this]() {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1}), sampler));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  m_textures.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&        gltfimage  = gltfModel.images[i];
    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = VkExtent2D{(uint32_t)gltfimage.width, (uint32_t)gltfimage.height};

    if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
    {
      addDefaultTexture();
      continue;
    }

    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures.emplace_back(m_alloc.createTexture(image, ivInfo, samplerCreateInfo));

    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)));
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkDestroyPipeline(m_device, m_waterSimAdvectionPipeline, nullptr);
  vkDestroyPipeline(m_device, m_waterSimGridNormalizedPipeline, nullptr);
  vkDestroyPipeline(m_device, m_waterSimDivFreePipeline, nullptr);
  vkDestroyPipeline(m_device, m_waterSimPressureSolvePipeline, nullptr);
  vkDestroyPipeline(m_device, m_waterSimProjectPipeline, nullptr);
  vkDestroyPipeline(m_device, m_waterSimTransferP2GPipeline0, nullptr);
  vkDestroyPipeline(m_device, m_waterSimTransferP2GPipeline1, nullptr);
  vkDestroyPipeline(m_device, m_waterSimTransferP2GPipeline2, nullptr);
  vkDestroyPipeline(m_device, m_waterSimTransferP2GPipeline3, nullptr);
  vkDestroyPipeline(m_device, m_waterSimTransferP2GPipeline4, nullptr);
  vkDestroyPipeline(m_device, m_drawParticlePipeline, nullptr);
  vkDestroyPipeline(m_device, m_drawWaterSurfacePipeline, nullptr);
  vkDestroyPipeline(m_device, m_marchingCubeMarkPipeline, nullptr);
  vkDestroyPipeline(m_device, m_marchingCubeScanPipeline, nullptr);
  vkDestroyPipeline(m_device, m_marchingCubeAssemblePipeline, nullptr);

  vkDestroyPipelineLayout(m_device, m_drawParticlePipelineLayout, nullptr);
  vkDestroyPipelineLayout(m_device, m_waterSimPipelineLayout, nullptr);
  vkDestroyPipelineLayout(m_device, m_marchingCubeMarkPipelineLayout, nullptr);
  vkDestroyPipelineLayout(m_device, m_marchingCubeScanPipelineLayout, nullptr);
  vkDestroyPipelineLayout(m_device, m_marchingCubeAssemblePipelineLayout, nullptr);

  vkDestroyDescriptorPool(m_device, m_waterSimParticleDescPool, nullptr);
  vkDestroyDescriptorPool(m_device, m_waterSimGridDescPool, nullptr);
  vkDestroyDescriptorPool(m_device, m_marchingCubeMarkDescPool, nullptr);
  vkDestroyDescriptorPool(m_device, m_marchingCubeScanDescPool, nullptr);
  vkDestroyDescriptorPool(m_device, m_marchingCubeAssemblyDescPool, nullptr);

  vkDestroyDescriptorSetLayout(m_device, m_waterSimParticleDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_waterSimGridDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_marchingCubeMarkDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_marchingCubeScanDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_marchingCubeAssemblyDescriptorSetLayout, nullptr);

  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_alloc.destroy(m_marchingCubeConfigBuffer);
  m_alloc.destroy(m_marchingCubeOffsetBuffer);
  m_alloc.destroy(m_marchingCubeVertexBuffer);
  m_alloc.destroy(m_marchingCubeIndexBuffer);
  m_alloc.destroy(m_marchingCubeNormalBuffer);
  m_alloc.destroy(m_marchingCubeGeomDescBuffer);
  m_alloc.destroy(m_marchingCubeScanOutputBuffer);
  m_alloc.destroy(m_marchingCubeScanStateBuffer);
  m_alloc.destroy(m_waterSurfaceAccelIndirectBuildRangeInfoBuffer);
  m_alloc.destroy(m_waterSurfaceAccelScratchBuffer);
  m_alloc.destroy(m_waterSurfaceAccel);

  m_alloc.destroy(m_picparticleBuffer);
  m_alloc.destroy(m_grid_v_x);
  m_alloc.destroy(m_grid_v_y);
  m_alloc.destroy(m_grid_v_z);
  m_alloc.destroy(m_grid_pressure);
  m_alloc.destroy(m_grid_weight);
  m_alloc.destroy(m_grid_marker);

  m_alloc.destroy(m_bGlobals);

  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primInfo);
  m_alloc.destroy(m_sceneDesc);

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

  // #VKRay
  m_rtBuilder.destroy();
  m_sbtWrapper.destroy();
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);


  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf)
{
  using vkPBP = VkPipelineBindPoint;

  std::vector<VkDeviceSize> offsets = {0, 0, 0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

  std::vector<VkBuffer> vertexBuffers = {m_vertexBuffer.buffer, m_normalBuffer.buffer, m_uvBuffer.buffer};
  vkCmdBindVertexBuffers(cmdBuf, 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), offsets.data());
  vkCmdBindIndexBuffer(cmdBuf, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

  uint32_t idxNode = 0;
  for(auto& node : m_gltfScene.m_nodes)
  {
    auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];

    m_pcRaster.modelMatrix = node.worldMatrix;
    m_pcRaster.objIndex    = node.primMesh;
    m_pcRaster.materialId  = primitive.materialIndex;
    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdDrawIndexed(cmdBuf, primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
  }

  if (m_drawWaterSurface)
  {
    m_debug.insertLabel(cmdBuf, "Draw Water Surface");
    std::vector<VkBuffer> waterGeomBuffers = {m_marchingCubeVertexBuffer.buffer, m_marchingCubeNormalBuffer.buffer};
    vkCmdBindVertexBuffers(cmdBuf, 0, static_cast<uint32_t>(waterGeomBuffers.size()), waterGeomBuffers.data(), offsets.data());
    vkCmdBindIndexBuffer(cmdBuf, m_marchingCubeIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawWaterSurfacePipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);
    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdDrawIndexed(cmdBuf, m_waterSurfaceAccelBuildRangeInfoPtr->primitiveCount * 3, 1, 0, 0, 0);
  }
  else
  {
    m_debug.insertLabel(cmdBuf, "Draw Water Particles");
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawParticlePipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawParticlePipelineLayout, 0, 1, &m_descSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawParticlePipelineLayout, 1, 1,
                            &m_waterSimParticleDescSet, 0, nullptr);
    vkCmdDraw(cmdBuf, m_waterSimParticleCount, 1, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
  resetFrame();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image  = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }


  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  VkWriteDescriptorSet writeDescriptorSets = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSets, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
  m_sbtWrapper.setup(m_device, m_graphicsQueueIndex, &m_alloc, m_rtProperties);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
auto HelloVulkan::primitiveToVkGeometry(const nvh::GltfPrimMesh& prim)
{
  // BLAS builder requires raw device addresses.
  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);

  uint32_t maxPrimitiveCount = prim.indexCount / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = 12;
  // Describe index data (32-bit unsigned int)
  triangles.indexType               = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = indexAddress;
  // Indicate identity transform by setting transformData to null device pointer.
  //triangles.transformData = {};
  triangles.maxVertex = prim.vertexCount - 1;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;  // For AnyHit
  asGeom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = prim.vertexOffset;
  offset.primitiveCount  = maxPrimitiveCount;
  offset.primitiveOffset = prim.firstIndex * sizeof(uint32_t);
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_gltfScene.m_primMeshes.size());
  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    auto geo = primitiveToVkGeometry(primMesh);
    allBlas.push_back({geo});
  }
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createTopLevelAS()
{
  m_tlas.reserve(m_gltfScene.m_nodes.size());
  for(auto& node : m_gltfScene.m_nodes)
  {
    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(node.worldMatrix);
    rayInst.instanceCustomIndex            = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.primMesh);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                           = 0xFF;
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    m_tlas.emplace_back(rayInst);
  }

  VkAccelerationStructureInstanceKHR rayInst{};
  rayInst.transform                              = nvvk::toTransformMatrixKHR(glm::identity<mat4>());
  rayInst.instanceCustomIndex                    = 99;
  VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  addressInfo.accelerationStructure      = m_waterSurfaceAccel.accel;
  rayInst.accelerationStructureReference         = vkGetAccelerationStructureDeviceAddressKHR(m_device,&addressInfo);
  rayInst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  rayInst.mask                                   = 0xFF;
  rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
  m_tlas.emplace_back(rayInst);

  m_tlasFlag = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_rtBuilder.buildTlas(m_tlas, m_tlasFlag );
}

void HelloVulkan::updateTopLevelAS()
{
  m_rtBuilder.buildTlas(m_tlas, m_tlasFlag,true);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  // Top-level acceleration structure, usable by both the ray generation and the closest hit (to shoot shadow rays)
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  // TLAS
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR);  // Output image
  m_rtDescSetLayoutBind.addBinding(RtxBindings::ePrimLookup, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                   VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);  // Primitive info

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);

  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool     = m_rtDescPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts        = &m_rtDescSetLayout;
  vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);


  VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo  imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo primitiveInfoDesc{m_primInfo.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::ePrimLookup, &primitiveInfoDesc));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo);
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eShaderGroupCount
  };

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtrace.rgen.spv", true, defaultSearchPaths, true));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtrace.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;
  // Hit Group - Closest Hit
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtrace.rchit.spv", true, defaultSearchPaths, true));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;


  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  m_rtShaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  m_rtShaderGroups.push_back(group);


  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
  // two miss shader groups, and one hit group.
  rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups    = m_rtShaderGroups.data();

  // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
  // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
  // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
  // in the ray generation to avoid deep recursion.
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


  // Creating the SBT
  m_sbtWrapper.create(m_rtPipeline, rayPipelineInfo);


  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor)
{
  updateFrame();

  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_pcRay.clearColor     = clearColor;
  m_pcRay.lightPosition  = m_pcRaster.lightPosition;
  m_pcRay.lightIntensity = m_pcRaster.lightIntensity;
  m_pcRay.lightType      = m_pcRaster.lightType;


  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                     0, sizeof(PushConstantRay), &m_pcRay);


  auto& regions = m_sbtWrapper.getRegions();
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);


  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
void HelloVulkan::updateFrame()
{
  static glm::mat4 refCamMatrix;
  static float     refFov{CameraManip.getFov()};

  const auto& m   = CameraManip.getMatrix();
  const auto  fov = CameraManip.getFov();

  if(refCamMatrix != m || refFov != fov)
  {
    resetFrame();
    refCamMatrix = m;
    refFov       = fov;
  }
  m_pcRay.frame++;
}

void HelloVulkan::resetFrame()
{
  m_pcRay.frame = -1;
}

void HelloVulkan::createWaterSimulationResources()
{
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  auto              cmdBuf                   = genCmdBuf.createCommandBuffer();

  m_picparticleBuffer           = m_alloc.createBuffer(cmdBuf, m_init_picparticles,
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_picparticleBuffer.buffer, "Particles");

  VkExtent3D gridSize = {m_waterSimGridX, m_waterSimGridY, m_waterSimGridZ};
  auto       grid3DCreateInfo =
      nvvk::makeImage3DCreateInfo(gridSize, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  nvvk::Image           image  = m_alloc.createImage(grid3DCreateInfo);
  VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, grid3DCreateInfo);
  m_grid_v_x = m_alloc.createTexture(image, ivInfo, sampler);
  m_grid_v_x.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  m_debug.setObjectName(m_grid_v_x.image, "Velocity x");
  m_debug.setObjectName(m_grid_v_x.descriptor.imageView, "Velocity x");
  nvvk::cmdBarrierImageLayout(cmdBuf, m_grid_v_x.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  image  = m_alloc.createImage(grid3DCreateInfo);
  ivInfo = nvvk::makeImageViewCreateInfo(image.image, grid3DCreateInfo);
  m_grid_v_y                        = m_alloc.createTexture(image, ivInfo, sampler);
  m_grid_v_y.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  m_debug.setObjectName(m_grid_v_y.image, "Velocity y");
  m_debug.setObjectName(m_grid_v_y.descriptor.imageView, "Velocity y");
  nvvk::cmdBarrierImageLayout(cmdBuf, m_grid_v_y.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  image       = m_alloc.createImage(grid3DCreateInfo);
  ivInfo      = nvvk::makeImageViewCreateInfo(image.image, grid3DCreateInfo);
  m_grid_v_z                        = m_alloc.createTexture(image, ivInfo, sampler);
  m_grid_v_z.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  m_debug.setObjectName(m_grid_v_z.image, "Velocity z");
  m_debug.setObjectName(m_grid_v_z.descriptor.imageView, "Velocity z");
  nvvk::cmdBarrierImageLayout(cmdBuf, m_grid_v_z.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  image       = m_alloc.createImage(grid3DCreateInfo);
  ivInfo      = nvvk::makeImageViewCreateInfo(image.image, grid3DCreateInfo);
  m_grid_pressure                      = m_alloc.createTexture(image, ivInfo, sampler);
  m_grid_pressure.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  m_debug.setObjectName(m_grid_pressure.image, "Pressure");
  m_debug.setObjectName(m_grid_pressure.descriptor.imageView, "Pressure");
  nvvk::cmdBarrierImageLayout(cmdBuf, m_grid_pressure.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  image       = m_alloc.createImage(grid3DCreateInfo);
  ivInfo      = nvvk::makeImageViewCreateInfo(image.image, grid3DCreateInfo);
  m_grid_weight                        = m_alloc.createTexture(image, ivInfo, sampler);
  m_grid_weight.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  m_debug.setObjectName(m_grid_weight.image, "Weight");
  m_debug.setObjectName(m_grid_weight.descriptor.imageView, "Weight");
  nvvk::cmdBarrierImageLayout(cmdBuf, m_grid_weight.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  image       = m_alloc.createImage(grid3DCreateInfo);
  ivInfo      = nvvk::makeImageViewCreateInfo(image.image, grid3DCreateInfo);
  m_grid_marker                        = m_alloc.createTexture(image, ivInfo, sampler);
  m_grid_marker.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  m_debug.setObjectName(m_grid_marker.image, "Marker");
  m_debug.setObjectName(m_grid_marker.descriptor.imageView, "Marker");
  nvvk::cmdBarrierImageLayout(cmdBuf, m_grid_marker.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  genCmdBuf.submitAndWait(cmdBuf);
}

void HelloVulkan::createWaterSimulationDescriptorSet()
{
  auto& bind = m_waterSimParticleSetLayoutBind;
  // Particles
  bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT);

  m_waterSimParticleDescriptorSetLayout = bind.createLayout(m_device);
  m_waterSimParticleDescPool            = bind.createPool(m_device, 1);
  m_waterSimParticleDescSet = nvvk::allocateDescriptorSet(m_device, m_waterSimParticleDescPool, m_waterSimParticleDescriptorSetLayout);

  auto & bind2 = m_waterSimGridSetLayoutBind;
  // Velocity x
  bind2.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Velocity y
  bind2.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Velocity z
  bind2.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Pressure
  bind2.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Weight
  bind2.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // Marker
  bind2.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  m_waterSimGridDescriptorSetLayout = bind2.createLayout(m_device);
  m_waterSimGridDescPool            = bind2.createPool(m_device, 1);
  m_waterSimGridDescSet = nvvk::allocateDescriptorSet(m_device, m_waterSimGridDescPool, m_waterSimGridDescriptorSetLayout);
}

void HelloVulkan::updateWaterSimulationDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiParticleBuf{m_picparticleBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_waterSimParticleSetLayoutBind.makeWrite(m_waterSimParticleDescSet, 0, &dbiParticleBuf));
  VkDescriptorImageInfo dbiVxImg{};
  dbiVxImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiVxImg.imageView   = m_grid_v_x.descriptor.imageView;
  dbiVxImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_waterSimGridSetLayoutBind.makeWrite(m_waterSimGridDescSet, 0, &dbiVxImg));
  VkDescriptorImageInfo dbiVyImg{};
  dbiVyImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiVyImg.imageView   = m_grid_v_y.descriptor.imageView;
  dbiVyImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_waterSimGridSetLayoutBind.makeWrite(m_waterSimGridDescSet, 1, &dbiVyImg));
  VkDescriptorImageInfo dbiVzImg{};
  dbiVzImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiVzImg.imageView   = m_grid_v_z.descriptor.imageView;
  dbiVzImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_waterSimGridSetLayoutBind.makeWrite(m_waterSimGridDescSet, 2, &dbiVzImg));
  VkDescriptorImageInfo dbiPressureImg{};
  dbiPressureImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiPressureImg.imageView   = m_grid_pressure.descriptor.imageView;
  dbiPressureImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_waterSimGridSetLayoutBind.makeWrite(m_waterSimGridDescSet, 3, &dbiPressureImg));
  VkDescriptorImageInfo dbiWeightImg{};
  dbiWeightImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiWeightImg.imageView   = m_grid_weight.descriptor.imageView;
  dbiWeightImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_waterSimGridSetLayoutBind.makeWrite(m_waterSimGridDescSet, 4, &dbiWeightImg));
  VkDescriptorImageInfo dbiMarkerImg{};
  dbiMarkerImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiMarkerImg.imageView   = m_grid_marker.descriptor.imageView;
  dbiMarkerImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_waterSimGridSetLayoutBind.makeWrite(m_waterSimGridDescSet, 5, &dbiMarkerImg));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void HelloVulkan::createWaterSimulationComputePipelines()
{
  VkPushConstantRange        pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset     = 0;
  pushConstantRange.size       = sizeof(glm::vec4);
  VkPipelineLayoutCreateInfo layoutInfo{};
  VkDescriptorSetLayout      layouts[2] = {m_waterSimParticleDescriptorSetLayout, m_waterSimGridDescriptorSetLayout};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 2;
  layoutInfo.pSetLayouts = layouts;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &pushConstantRange;
  vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_waterSimPipelineLayout);

  std::vector<std::string> paths = defaultSearchPaths;
  
  VkPipelineShaderStageCreateInfo shaderInfo{};
  shaderInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderInfo.pName                  = "main";
  shaderInfo.stage                  = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderInfo.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/advection.comp.spv", true, defaultSearchPaths, true));
  
  VkComputePipelineCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  createInfo.layout = m_waterSimPipelineLayout;
  createInfo.stage  = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimAdvectionPipeline);
  m_debug.setObjectName(m_waterSimAdvectionPipeline, "waterSimAdvection");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  shaderInfo.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/grid_normalized.comp.spv", true, defaultSearchPaths, true));
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimGridNormalizedPipeline);
  m_debug.setObjectName(m_waterSimGridNormalizedPipeline, "waterSimGridNormalized");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  shaderInfo.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/div_free.comp.spv", true, defaultSearchPaths, true));
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimDivFreePipeline);
  m_debug.setObjectName(m_waterSimDivFreePipeline, "waterSimDivFree");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  shaderInfo.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/pressure_solve.comp.spv", true, defaultSearchPaths, true));
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimPressureSolvePipeline);
  m_debug.setObjectName(m_waterSimDivFreePipeline, "waterSimDivFree");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  shaderInfo.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/project.comp.spv", true, defaultSearchPaths, true));
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimProjectPipeline);
  m_debug.setObjectName(m_waterSimDivFreePipeline, "waterSimDivFree");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  int                      gridID      = 0;
  VkSpecializationMapEntry entry     = {0, 0, sizeof(int32_t)};
  VkSpecializationInfo     spec_info = {1, &entry, sizeof(uint32_t), &gridID};

  shaderInfo.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/transfer_p2g.comp.spv", true, defaultSearchPaths, true));
  shaderInfo.pSpecializationInfo = &spec_info;
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimTransferP2GPipeline0);
  m_debug.setObjectName(m_waterSimTransferP2GPipeline0, "waterSimTransferP2G0");
  //vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  gridID = 1;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimTransferP2GPipeline1);
  m_debug.setObjectName(m_waterSimTransferP2GPipeline1, "waterSimTransferP2G1");
  //vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  gridID = 2;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimTransferP2GPipeline2);
  m_debug.setObjectName(m_waterSimTransferP2GPipeline2, "waterSimTransferP2G2");
  //vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  gridID = 3;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimTransferP2GPipeline3);
  m_debug.setObjectName(m_waterSimTransferP2GPipeline3, "waterSimTransferP2G3");
  //vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  gridID = 4;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_waterSimTransferP2GPipeline4);
  m_debug.setObjectName(m_waterSimTransferP2GPipeline4, "waterSimTransferP2G4");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);
}

void HelloVulkan::waterSimStep(const VkCommandBuffer& cmdBuff) 
{
  static VkImageSubresourceRange range;
  range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  range.baseArrayLayer = 0;
  range.baseMipLevel   = 0;
  range.layerCount     = 1;
  range.levelCount     = 1;

  const int            textureCount = 6;
  VkImageMemoryBarrier gridBarriers[textureCount];
  for(int i = 0; i < textureCount; i++)
  {
    gridBarriers[i].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    gridBarriers[i].pNext               = nullptr;
    gridBarriers[i].subresourceRange    = range;
    gridBarriers[i].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    gridBarriers[i].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    gridBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    gridBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    gridBarriers[i].srcQueueFamilyIndex = 0;
    gridBarriers[i].dstQueueFamilyIndex = 0;
  }
  gridBarriers[0].image = m_grid_v_x.image;
  gridBarriers[1].image = m_grid_v_y.image;
  gridBarriers[2].image = m_grid_v_z.image;
  gridBarriers[3].image = m_grid_pressure.image;
  gridBarriers[4].image = m_grid_weight.image;
  gridBarriers[5].image = m_grid_marker.image;

  m_debug.beginLabel(cmdBuff ,"waterSimStep");
 
  vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimPipelineLayout, 0, 1, &m_waterSimParticleDescSet,0,nullptr);
  vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimPipelineLayout, 1, 1, &m_waterSimGridDescSet,0,nullptr);
  vkCmdPushConstants(cmdBuff, m_waterSimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_waterSimPushConstant),
                     &m_waterSimPushConstant);

  m_debug.insertLabel(cmdBuff, "advection");

  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimAdvectionPipeline);
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, textureCount, gridBarriers);
  vkCmdDispatch(cmdBuff, m_waterSimParticleCount/128 , 1, 1);
  
  m_debug.insertLabel(cmdBuff, "clear grid");
  
  for(int i = 0; i < textureCount; i++)
  {
    gridBarriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    gridBarriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  }

  static VkClearColorValue black{0,0,0,0};
  
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, textureCount, gridBarriers);

  vkCmdClearColorImage(cmdBuff, m_grid_v_x.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
  vkCmdClearColorImage(cmdBuff, m_grid_v_y.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
  vkCmdClearColorImage(cmdBuff, m_grid_v_z.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
  // We don't clear the old pressure value since it might be a good initial guess
  //vkCmdClearColorImage(cmdBuff, m_grid_pressure.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
  vkCmdClearColorImage(cmdBuff, m_grid_weight.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
  vkCmdClearColorImage(cmdBuff, m_grid_marker.image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);

  // Transfer Particles to Grids
  m_debug.insertLabel(cmdBuff, "transfer p2g");
  VkBufferMemoryBarrier memBarrier1{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  memBarrier1.pNext               = nullptr;
  memBarrier1.buffer        = m_picparticleBuffer.buffer;
  memBarrier1.offset        = 0;
  memBarrier1.size          = sizeof(PICParticle) * m_init_picparticles.size();
  memBarrier1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  memBarrier1.srcQueueFamilyIndex = 0;
  memBarrier1.dstQueueFamilyIndex = 0;
  // Make sure particle advection and grid clear has finished
  for(int i = 0; i < 6; i++)
  {
    gridBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    gridBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &memBarrier1, textureCount, gridBarriers);
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimTransferP2GPipeline0);
  vkCmdDispatch(cmdBuff, m_waterSimParticleCount / 16, 1, 1);
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimTransferP2GPipeline1);
  vkCmdDispatch(cmdBuff, m_waterSimParticleCount / 16, 1, 1);
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimTransferP2GPipeline2);
  vkCmdDispatch(cmdBuff, m_waterSimParticleCount / 16, 1, 1);
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimTransferP2GPipeline3);
  vkCmdDispatch(cmdBuff, m_waterSimParticleCount / 16, 1, 1);
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimTransferP2GPipeline4);
  vkCmdDispatch(cmdBuff, m_waterSimParticleCount / 16, 1, 1);

  m_debug.insertLabel(cmdBuff, "grid normalize");
  // Make sure transfer has finished
  for(int i = 0; i < textureCount; i++)
  {
    gridBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    gridBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }

  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimGridNormalizedPipeline);
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, textureCount, gridBarriers);
  vkCmdDispatch(cmdBuff, m_waterSimGridX/8, m_waterSimGridY/8, m_waterSimGridZ/8);

  m_debug.beginLabel(cmdBuff, "pressure solve jacobi");
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimPressureSolvePipeline);
  for (int i = 0; i < 16; i++)
  {
    vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, textureCount, gridBarriers);
    vkCmdPushConstants(cmdBuff, m_waterSimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(m_waterSimPushConstant), &m_waterSimPushConstant);
    vkCmdDispatch(cmdBuff, m_waterSimGridX/ 8, m_waterSimGridY/8, m_waterSimGridZ/8);
  }
  m_debug.endLabel(cmdBuff);
  m_debug.insertLabel(cmdBuff, "projection");
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, textureCount, gridBarriers);
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_waterSimProjectPipeline);
  vkCmdDispatch(cmdBuff, m_waterSimGridX / 8, m_waterSimGridY / 8, m_waterSimGridZ / 8);

  m_debug.endLabel(cmdBuff);
}

void HelloVulkan::createWaterSurfaceReconstructResources(){

  uint32_t maxElementCount = (m_waterSimGridX - 1) * (m_waterSimGridY - 1) * (m_waterSimGridZ - 1);

  m_marchingCubeConfigBuffer = m_alloc.createBuffer(maxElementCount * sizeof(int),
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeConfigBuffer.buffer, "MarchingCubeConfigBuffer");

  m_marchingCubeOffsetBuffer = m_alloc.createBuffer(maxElementCount * sizeof(glm::ivec2),
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeOffsetBuffer.buffer, "MarchingCubeOffsetBuffer");

  m_marchingCubeScanOutputBuffer = m_alloc.createBuffer(maxElementCount * sizeof(glm::ivec2), 
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeOffsetBuffer.buffer, "MarchingCubeScanOutputBuffer");

  const uint ELEMENTS_PER_WG         = 512 * 16;
  uint n_workgroups = (maxElementCount + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
  m_marchingCubeScanStateBuffer = m_alloc.createBuffer(n_workgroups * 36, 
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeOffsetBuffer.buffer, "MarchingCubeScanStateBuffer");

  uint32_t maxElementForMarchingCube = 10 * (m_waterSimGridX - 1) * (m_waterSimGridY - 1);

  m_marchingCubeIndexBuffer = m_alloc.createBuffer(maxElementForMarchingCube * 16 * sizeof(uint),
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeIndexBuffer.buffer, "MarchingCubeIndexBuffer");

  m_marchingCubeVertexBuffer = m_alloc.createBuffer(maxElementForMarchingCube * 12 * sizeof(vec3),
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeVertexBuffer.buffer, "MarchingCubeVertexBuffer");

  m_marchingCubeNormalBuffer =
      m_alloc.createBuffer(maxElementForMarchingCube * 12 * sizeof(vec3),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_marchingCubeNormalBuffer.buffer, "MarchingCubeNormalBuffer");

  std::vector<WaterSurfaceGeomDesc> desc(1);
  desc[0].vertexAddress = nvvk::getBufferDeviceAddress(m_device,m_marchingCubeVertexBuffer.buffer);
  desc[0].indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_marchingCubeIndexBuffer.buffer);
  desc[0].normalAddress = nvvk::getBufferDeviceAddress(m_device, m_marchingCubeNormalBuffer.buffer);

  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();
  m_marchingCubeGeomDescBuffer = m_alloc.createBuffer(cmdBuf, desc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  genCmdBuf.submitAndWait(cmdBuf);
  m_debug.setObjectName(m_marchingCubeGeomDescBuffer.buffer, "MarchingCubeGeomDescBuffer");

  m_waterSurfaceAccelIndirectBuildRangeInfoBuffer =
      m_alloc.createBuffer(sizeof(VkAccelerationStructureBuildRangeInfoKHR),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  m_debug.setObjectName(m_waterSurfaceAccelIndirectBuildRangeInfoBuffer.buffer,
                        "WaterSurfaceAccelIndirectBuildRangeInfo");

  m_waterSurfaceAccelBuildRangeInfoPtr = static_cast<VkAccelerationStructureBuildRangeInfoKHR*>(m_alloc.map(m_waterSurfaceAccelIndirectBuildRangeInfoBuffer));

  // Create BLAS for water surface
  m_waterSurfaceAccelTrianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  m_waterSurfaceAccelTrianglesData.pNext        = nullptr;
  m_waterSurfaceAccelTrianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  m_waterSurfaceAccelTrianglesData.vertexData =
      VkDeviceOrHostAddressConstKHR{nvvk::getBufferDeviceAddressKHR(m_device, m_marchingCubeVertexBuffer.buffer)};
  m_waterSurfaceAccelTrianglesData.vertexStride = sizeof(vec3);
  m_waterSurfaceAccelTrianglesData.maxVertex    = maxElementCount;
  m_waterSurfaceAccelTrianglesData.indexType    = VK_INDEX_TYPE_UINT32;
  m_waterSurfaceAccelTrianglesData.indexData =
      VkDeviceOrHostAddressConstKHR{nvvk::getBufferDeviceAddress(m_device, m_marchingCubeIndexBuffer.buffer)};

  m_waterSurfaceAccelGeometry.sType = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  m_waterSurfaceAccelGeometry.pNext        = nullptr;
  m_waterSurfaceAccelGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  m_waterSurfaceAccelGeometry.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
  m_waterSurfaceAccelGeometry.geometry     = VkAccelerationStructureGeometryDataKHR{m_waterSurfaceAccelTrianglesData};

  m_waterSurfaceAccelBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  m_waterSurfaceAccelBuildGeometryInfo.pNext = nullptr;
  m_waterSurfaceAccelBuildGeometryInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  m_waterSurfaceAccelBuildGeometryInfo.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
  m_waterSurfaceAccelBuildGeometryInfo.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  m_waterSurfaceAccelBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  m_waterSurfaceAccelBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
  m_waterSurfaceAccelBuildGeometryInfo.geometryCount            = 1;
  m_waterSurfaceAccelBuildGeometryInfo.pGeometries              = &m_waterSurfaceAccelGeometry;
  m_waterSurfaceAccelBuildGeometryInfo.ppGeometries             = nullptr;
  m_waterSurfaceAccelBuildGeometryInfo.scratchData              = VkDeviceOrHostAddressKHR{0};

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  const uint32_t                           max_primitive_counts[1] = {maxElementCount};

  vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &m_waterSurfaceAccelBuildGeometryInfo,
                                          max_primitive_counts, &sizeInfo);

  VkAccelerationStructureCreateInfoKHR accelCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  accelCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  accelCreateInfo.offset = 0;
  accelCreateInfo.size   = sizeInfo.accelerationStructureSize;
  
  m_waterSurfaceAccel =  m_alloc.createAcceleration(accelCreateInfo);

  m_debug.setObjectName(m_waterSurfaceAccel.accel, "Water Surface");

  m_waterSurfaceAccelScratchBuffer =
      m_alloc.createBuffer(sizeInfo.buildScratchSize, 
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                              | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  m_debug.setObjectName(m_waterSurfaceAccelScratchBuffer.buffer, "Water Surface Accel Scratch");

  m_waterSurfaceAccelBuildGeometryInfo.dstAccelerationStructure = m_waterSurfaceAccel.accel;
  m_waterSurfaceAccelBuildGeometryInfo.scratchData = VkDeviceOrHostAddressKHR{nvvk::getBufferDeviceAddress(m_device,m_waterSurfaceAccelScratchBuffer.buffer)};
};

void HelloVulkan::createWaterSurfaceReconstructDescriptorSet()
{
  m_marchingCubeMarkSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeMarkSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeMarkSetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeMarkDescriptorSetLayout = m_marchingCubeMarkSetLayoutBind.createLayout(m_device);
  m_marchingCubeMarkDescPool            = m_marchingCubeMarkSetLayoutBind.createPool(m_device, 1);
  m_marchingCubeMarkDescSet =
      nvvk::allocateDescriptorSet(m_device, m_marchingCubeMarkDescPool, m_marchingCubeMarkDescriptorSetLayout);

  m_marchingCubeScanSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeScanSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeScanSetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeScanDescriptorSetLayout = m_marchingCubeScanSetLayoutBind.createLayout(m_device);
  m_marchingCubeScanDescPool            = m_marchingCubeScanSetLayoutBind.createPool(m_device, 1);
  m_marchingCubeScanDescSet =
      nvvk::allocateDescriptorSet(m_device, m_marchingCubeScanDescPool, m_marchingCubeScanDescriptorSetLayout);

  m_marchingCubeAssemblySetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeAssemblySetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeAssemblySetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeAssemblySetLayoutBind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_marchingCubeAssemblyDescriptorSetLayout = m_marchingCubeAssemblySetLayoutBind.createLayout(m_device);
  m_marchingCubeAssemblyDescPool            = m_marchingCubeAssemblySetLayoutBind.createPool(m_device, 1);
  m_marchingCubeAssemblyDescSet =
      nvvk::allocateDescriptorSet(m_device, m_marchingCubeAssemblyDescPool, m_marchingCubeAssemblyDescriptorSetLayout);
}

void HelloVulkan::updateWaterSurfaceReconstructDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;
  //Update MarchingCubeMarkDescSet
  VkDescriptorImageInfo dbiWeightImg{};
  dbiWeightImg.imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL;
  dbiWeightImg.imageView   = m_grid_weight.descriptor.imageView;
  dbiWeightImg.sampler     = VK_NULL_HANDLE;
  writes.emplace_back(m_marchingCubeMarkSetLayoutBind.makeWrite(m_marchingCubeMarkDescSet, 0, &dbiWeightImg));
  const VkDescriptorBufferInfo dbiMarchingCubeConfigBuf{m_marchingCubeConfigBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_marchingCubeMarkSetLayoutBind.makeWrite(m_marchingCubeMarkDescSet, 1, &dbiMarchingCubeConfigBuf));
  const VkDescriptorBufferInfo dbiMarchingCubeOffsetBuf{m_marchingCubeOffsetBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_marchingCubeMarkSetLayoutBind.makeWrite(m_marchingCubeMarkDescSet, 2, &dbiMarchingCubeOffsetBuf));

  //Update MarchingCubeScanDescSet
  writes.emplace_back(m_marchingCubeScanSetLayoutBind.makeWrite(m_marchingCubeScanDescSet, 0, &dbiMarchingCubeOffsetBuf));
  const VkDescriptorBufferInfo dbiMarchingCubeScanStateBuf{m_marchingCubeScanStateBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_marchingCubeScanSetLayoutBind.makeWrite(m_marchingCubeScanDescSet, 1, &dbiMarchingCubeScanStateBuf));
  const VkDescriptorBufferInfo dbiMarchingCubeScanOutputBuf{m_marchingCubeScanOutputBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_marchingCubeScanSetLayoutBind.makeWrite(m_marchingCubeScanDescSet, 2, &dbiMarchingCubeScanOutputBuf));

  //Update MarchingCubeAssemblyDescSet
  writes.emplace_back(m_marchingCubeAssemblySetLayoutBind.makeWrite(m_marchingCubeAssemblyDescSet, 0, &dbiMarchingCubeConfigBuf));
  writes.emplace_back(m_marchingCubeAssemblySetLayoutBind.makeWrite(m_marchingCubeAssemblyDescSet, 1, &dbiMarchingCubeScanOutputBuf));
  const VkDescriptorBufferInfo dbiMarchingCubeGeomDescBuf{m_marchingCubeGeomDescBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_marchingCubeAssemblySetLayoutBind.makeWrite(m_marchingCubeAssemblyDescSet, 2, &dbiMarchingCubeGeomDescBuf));
  const VkDescriptorBufferInfo dbiMarchingAccelRangeInfoBuf{m_waterSurfaceAccelIndirectBuildRangeInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_marchingCubeAssemblySetLayoutBind.makeWrite(m_marchingCubeAssemblyDescSet, 3, &dbiMarchingAccelRangeInfoBuf));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void HelloVulkan::createWaterSurfaceReconstructComputePipelines()
{
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset     = 0;
  pushConstantRange.size       = sizeof(glm::vec4);
  VkPipelineLayoutCreateInfo layoutInfo{};
  VkDescriptorSetLayout      layouts[1] = {m_marchingCubeMarkDescriptorSetLayout};
  layoutInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount             = 1;
  layoutInfo.pSetLayouts                = layouts;
  layoutInfo.pushConstantRangeCount     = 1;
  layoutInfo.pPushConstantRanges        = &pushConstantRange;
  vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_marchingCubeMarkPipelineLayout);

  layoutInfo.pSetLayouts = &m_marchingCubeScanDescriptorSetLayout;
  layoutInfo.setLayoutCount = 1;
  vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_marchingCubeScanPipelineLayout);

  layoutInfo.pSetLayouts = &m_marchingCubeAssemblyDescriptorSetLayout;
  vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_marchingCubeAssemblePipelineLayout);

  std::vector<std::string> paths = defaultSearchPaths;

  VkPipelineShaderStageCreateInfo shaderInfo{};
  shaderInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderInfo.pName = "main";
  shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderInfo.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/marching_cube.comp.spv", true, defaultSearchPaths, true));

  VkComputePipelineCreateInfo createInfo{};
  createInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  createInfo.layout = m_marchingCubeMarkPipelineLayout;
  createInfo.stage  = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_marchingCubeMarkPipeline);
  m_debug.setObjectName(m_marchingCubeMarkPipeline, "marchingCubeMark");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  createInfo.layout = m_marchingCubeScanPipelineLayout;
  shaderInfo.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/prefix_sum.comp.spv", true, defaultSearchPaths, true));
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_marchingCubeScanPipeline);
  m_debug.setObjectName(m_marchingCubeScanPipeline, "marchingCubeScan");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);

  createInfo.layout = m_marchingCubeAssemblePipelineLayout;
  shaderInfo.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/marching_cube_assemble.comp.spv", true, defaultSearchPaths, true));
  createInfo.stage = shaderInfo;
  vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_marchingCubeAssemblePipeline);
  m_debug.setObjectName(m_marchingCubeAssemblePipeline, "marchingCubeAssemble");
  vkDestroyShaderModule(m_device, shaderInfo.module, nullptr);
}

void HelloVulkan::waterSurfaceReconstruct()
{
  nvvk::CommandPool              genCmdBuf(m_device, m_graphicsQueueIndex);
  VkCommandBuffer                cmdBuff = genCmdBuf.createCommandBuffer();

  static VkImageSubresourceRange range;
  range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  range.baseArrayLayer = 0;
  range.baseMipLevel   = 0;
  range.layerCount     = 1;
  range.levelCount     = 1;

  const int            textureCount = 6;
  VkImageMemoryBarrier gridBarriers[textureCount];
  for(int i = 0; i < textureCount; i++)
  {
    gridBarriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    gridBarriers[i].pNext               = nullptr;
    gridBarriers[i].subresourceRange    = range;
    gridBarriers[i].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    gridBarriers[i].oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    gridBarriers[i].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    gridBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    gridBarriers[i].srcQueueFamilyIndex = 0;
    gridBarriers[i].dstQueueFamilyIndex = 0;
  }
  gridBarriers[0].image = m_grid_v_x.image;
  gridBarriers[1].image = m_grid_v_y.image;
  gridBarriers[2].image = m_grid_v_z.image;
  gridBarriers[3].image = m_grid_pressure.image;
  gridBarriers[4].image = m_grid_weight.image;
  gridBarriers[5].image = m_grid_marker.image;

  VkBufferMemoryBarrier memBarriers[4];

  for (int i = 0; i < 4; i++)
  {
    memBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    memBarriers[i].pNext             = nullptr;
    memBarriers[i].offset           = 0;
    memBarriers[i].size              = VK_WHOLE_SIZE;
    memBarriers[i].srcAccessMask     = VK_ACCESS_SHADER_WRITE_BIT;
    memBarriers[i].dstAccessMask     = VK_ACCESS_SHADER_READ_BIT;
    memBarriers[i].srcQueueFamilyIndex = 0;
    memBarriers[i].dstQueueFamilyIndex = 0;
  }

  memBarriers[0].buffer = m_marchingCubeConfigBuffer.buffer;
  memBarriers[1].buffer = m_marchingCubeOffsetBuffer.buffer;
  memBarriers[2].buffer = m_marchingCubeScanStateBuffer.buffer;
  memBarriers[3].buffer = m_marchingCubeScanOutputBuffer.buffer;

  m_debug.beginLabel(cmdBuff, "water surface reconstruction");
  m_debug.insertLabel(cmdBuff, "mark marching cube");
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_marchingCubeMarkPipeline);
  vkCmdPushConstants(cmdBuff, m_marchingCubeMarkPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_waterSimPushConstant),
                     &m_waterSimPushConstant);
  vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_marchingCubeMarkPipelineLayout, 0, 1,
                          &m_marchingCubeMarkDescSet, 0, nullptr);
  // Make sure water simulation has finished
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, textureCount, gridBarriers);
  vkCmdDispatch(cmdBuff, m_waterSimGridX / 8, m_waterSimGridY / 8, m_waterSimGridZ / 8);

  m_debug.beginLabel(cmdBuff, "prefix sum");
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_marchingCubeScanPipeline);
  vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_marchingCubeScanPipelineLayout, 0, 1,
                          &m_marchingCubeScanDescSet, 0, nullptr);
  vkCmdFillBuffer(cmdBuff, m_marchingCubeScanStateBuffer.buffer, 0, VK_WHOLE_SIZE, 0);
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 2, &memBarriers[1], 0, nullptr);
  const uint maxElementCount = (m_waterSimGridX - 1) * (m_waterSimGridY - 1) * (m_waterSimGridZ - 1);
  const uint ELEMENTS_PER_WG = 512 * 16;
  const uint       n_workgroups    = (maxElementCount + ELEMENTS_PER_WG - 1) / ELEMENTS_PER_WG;
  vkCmdDispatch(cmdBuff, n_workgroups, 1, 1);
  m_debug.endLabel(cmdBuff);

  m_debug.insertLabel(cmdBuff, "marching cube assemble");
  vkCmdBindPipeline(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_marchingCubeAssemblePipeline);
  vkCmdPushConstants(cmdBuff, m_marchingCubeAssemblePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(m_waterSimPushConstant),
                     &m_waterSimPushConstant);
  vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_marchingCubeAssemblePipelineLayout, 0, 1,
                          &m_marchingCubeAssemblyDescSet, 0, nullptr);
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 4, memBarriers, 0, nullptr);
  vkCmdDispatch(cmdBuff, m_waterSimGridX / 8, m_waterSimGridY / 8, m_waterSimGridZ / 8);

  m_debug.endLabel(cmdBuff);

  genCmdBuf.submitAndWait(cmdBuff);
}

void HelloVulkan::rebuildWaterSurfaceBLAS()
{
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();

  m_debug.beginLabel(cmdBuf, "build BLAS for water surface");

  VkBufferMemoryBarrier memBarriers[2];
  for(int i = 0; i < 2; i++)
  {
    memBarriers[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    memBarriers[i].pNext               = nullptr;
    memBarriers[i].offset              = 0;
    memBarriers[i].size                = VK_WHOLE_SIZE;
    memBarriers[i].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    memBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    memBarriers[i].srcQueueFamilyIndex = 0;
    memBarriers[i].dstQueueFamilyIndex = 0;
  }

  memBarriers[0].buffer = m_marchingCubeVertexBuffer.buffer;
  memBarriers[1].buffer = m_marchingCubeIndexBuffer.buffer;

  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 2, memBarriers, 0, nullptr);
  uint32_t indirectStride = 0;
  uint32_t  max_primitive_counts[1] = {m_waterSimGridX * m_waterSimGridY * m_waterSimGridZ};
  uint32_t*      pMax_primitive_counts   = max_primitive_counts;
  const VkDeviceAddress rangeInfoAddr =
      nvvk::getBufferDeviceAddress(m_device, m_waterSurfaceAccelIndirectBuildRangeInfoBuffer.buffer);
  m_waterSurfaceAccelBuildRangeInfoPtr->firstVertex = 0;
  m_waterSurfaceAccelBuildRangeInfoPtr->primitiveOffset = 0;
  m_waterSurfaceAccelBuildRangeInfoPtr->transformOffset = 0;
  vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &m_waterSurfaceAccelBuildGeometryInfo,&m_waterSurfaceAccelBuildRangeInfoPtr);
  m_debug.endLabel(cmdBuf);
  genCmdBuf.submitAndWait(cmdBuf);
}

void HelloVulkan::rebuildWaterSurfaceBLASIndirect(const VkCommandBuffer& cmdBuff)
{
  m_debug.beginLabel(cmdBuff, "indirect build BLAS for water surface");

  VkBufferMemoryBarrier memBarriers[3];
  for(int i = 0; i < 3; i++)
  {
    memBarriers[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    memBarriers[i].pNext               = nullptr;
    memBarriers[i].offset              = 0;
    memBarriers[i].size                = VK_WHOLE_SIZE;
    memBarriers[i].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    memBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    memBarriers[i].srcQueueFamilyIndex = 0;
    memBarriers[i].dstQueueFamilyIndex = 0;
  }

  memBarriers[0].buffer = m_marchingCubeVertexBuffer.buffer;
  memBarriers[1].buffer = m_marchingCubeIndexBuffer.buffer;
  memBarriers[2].buffer = m_waterSurfaceAccelIndirectBuildRangeInfoBuffer.buffer;
  vkCmdPipelineBarrier(cmdBuff, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 3, memBarriers, 0, nullptr);
  uint32_t              indirectStirde          = 0;
  uint32_t              max_primitive_counts[1] = {m_waterSimGridX * m_waterSimGridY * m_waterSimGridZ};
  uint32_t*             pMax_primitive_counts   = max_primitive_counts;
  const VkDeviceAddress rangeInfoAddr =
      nvvk::getBufferDeviceAddress(m_device, m_waterSurfaceAccelIndirectBuildRangeInfoBuffer.buffer);
  vkCmdBuildAccelerationStructuresIndirectKHR(cmdBuff, 1, &m_waterSurfaceAccelBuildGeometryInfo, &rangeInfoAddr,
                                              &indirectStirde, &pMax_primitive_counts);
  m_debug.endLabel(cmdBuff);
}