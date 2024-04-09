/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "shaders/host_device.h"

#include "nvvkhl/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"

// #VKRay
#include "nvh/gltfscene.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvkhl::AppBaseVk
{
public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadScene(const std::string& filename);
  void updateDescriptorSet();
  void createUniformBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);

  void createWaterSimulationResources();
  void createWaterSimulationDescriptorSet();
  void updateWaterSimulationDescriptorSet();
  void createWaterSimulationComputePipelines();
  void createWaterSurfaceReconstructResources();
  void createWaterSurfaceReconstructDescriptorSet();
  void updateWaterSurfaceReconstructDescriptorSet();
  void createWaterSurfaceReconstructComputePipelines();
  void waterSimStep(const VkCommandBuffer& cmdBuff);
  void waterSurfaceReconstruct();
  void rebuildWaterSurfaceBLAS();
  void rebuildWaterSurfaceBLASIndirect(const VkCommandBuffer& cmdBuff);

  VkPipelineLayout m_waterSimPipelineLayout;
  VkPipelineLayout m_drawParticlePipelineLayout;

  VkPipeline m_waterSimAdvectionPipeline;
  VkPipeline m_waterSimGridNormalizedPipeline;
  VkPipeline m_waterSimDivFreePipeline;
  VkPipeline m_waterSimPressureSolvePipeline;
  VkPipeline m_waterSimProjectPipeline;
  VkPipeline m_waterSimTransferP2GPipeline0;
  VkPipeline m_waterSimTransferP2GPipeline1;
  VkPipeline m_waterSimTransferP2GPipeline2;
  VkPipeline m_waterSimTransferP2GPipeline3;
  VkPipeline m_waterSimTransferP2GPipeline4;
  VkPipeline m_drawParticlePipeline;
  VkPipeline m_drawWaterSurfacePipeline;

  bool m_drawWaterSurface = false;

  nvvk::DescriptorSetBindings m_waterSimParticleSetLayoutBind;
  VkDescriptorPool            m_waterSimParticleDescPool;
  VkDescriptorSetLayout       m_waterSimParticleDescriptorSetLayout;
  VkDescriptorSet             m_waterSimParticleDescSet;

  nvvk::DescriptorSetBindings m_waterSimGridSetLayoutBind;
  VkDescriptorPool            m_waterSimGridDescPool;
  VkDescriptorSetLayout       m_waterSimGridDescriptorSetLayout;
  VkDescriptorSet             m_waterSimGridDescSet;

  uint32_t  m_waterSimParticleCount;
  uint32_t  m_waterSimGridX;
  uint32_t  m_waterSimGridY;
  uint32_t  m_waterSimGridZ;
  glm::vec4 m_waterSimPushConstant;

  // Water simulation Data
  std::vector<PICParticle> m_init_picparticles;
  nvvk::Buffer             m_picparticleBuffer;
  nvvk::Texture            m_grid_v_x;
  nvvk::Texture            m_grid_v_y;
  nvvk::Texture           m_grid_v_z;
  nvvk::Texture            m_grid_pressure;
  nvvk::Texture           m_grid_weight;
  nvvk::Texture            m_grid_marker;
  
  nvh::GltfScene m_gltfScene;
  nvvk::Buffer   m_vertexBuffer;
  nvvk::Buffer   m_normalBuffer;
  nvvk::Buffer   m_uvBuffer;
  nvvk::Buffer   m_indexBuffer;
  nvvk::Buffer   m_materialBuffer;
  nvvk::Buffer   m_primInfo;
  nvvk::Buffer   m_sceneDesc;

  VkPipelineLayout m_marchingCubeMarkPipelineLayout;
  VkPipelineLayout m_marchingCubeScanPipelineLayout;
  VkPipelineLayout m_marchingCubeAssemblePipelineLayout;

  VkPipeline m_marchingCubeMarkPipeline;
  VkPipeline m_marchingCubeScanPipeline;
  VkPipeline m_marchingCubeAssemblePipeline;

  nvvk::DescriptorSetBindings m_marchingCubeMarkSetLayoutBind;
  VkDescriptorPool            m_marchingCubeMarkDescPool;
  VkDescriptorSetLayout       m_marchingCubeMarkDescriptorSetLayout;
  VkDescriptorSet             m_marchingCubeMarkDescSet;

  nvvk::DescriptorSetBindings m_marchingCubeScanSetLayoutBind;
  VkDescriptorPool            m_marchingCubeScanDescPool;
  VkDescriptorSetLayout       m_marchingCubeScanDescriptorSetLayout;
  VkDescriptorSet             m_marchingCubeScanDescSet;

  nvvk::DescriptorSetBindings m_marchingCubeAssemblySetLayoutBind;
  VkDescriptorPool            m_marchingCubeAssemblyDescPool;
  VkDescriptorSetLayout       m_marchingCubeAssemblyDescriptorSetLayout;
  VkDescriptorSet             m_marchingCubeAssemblyDescSet;
    
  VkSampler    weightSampler;
  nvvk::Buffer m_marchingCubeConfigBuffer;
  nvvk::Buffer m_marchingCubeOffsetBuffer;
  nvvk::Buffer m_marchingCubeScanOutputBuffer;
  nvvk::Buffer m_marchingCubeScanStateBuffer;
  nvvk::Buffer m_marchingCubeVertexBuffer;
  nvvk::Buffer m_marchingCubeIndexBuffer;
  nvvk::Buffer m_marchingCubeNormalBuffer;
  nvvk::Buffer m_marchingCubeGeomDescBuffer;

  VkAccelerationStructureGeometryTrianglesDataKHR m_waterSurfaceAccelTrianglesData{};
  VkAccelerationStructureGeometryKHR              m_waterSurfaceAccelGeometry{};
  VkAccelerationStructureBuildGeometryInfoKHR     m_waterSurfaceAccelBuildGeometryInfo{};
  nvvk::AccelKHR m_waterSurfaceAccel;
  nvvk::Buffer                                    m_waterSurfaceAccelScratchBuffer;
  nvvk::Buffer                                    m_waterSurfaceAccelIndirectBuildRangeInfoBuffer;
  VkAccelerationStructureBuildRangeInfoKHR*        m_waterSurfaceAccelBuildRangeInfoPtr;

  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},  // Identity matrix
      {0.f, 4.5f, 0.f},                                  // light position
      0,                                                 // instance Id
      10.f,                                              // light intensity
      0,                                                 // light type
      0                                                  // material id
  };

  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer               m_bGlobals;  // Device-Host of the camera matrices
  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene

  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects

  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};

  // #VKRay
  void initRayTracing();
  auto primitiveToVkGeometry(const nvh::GltfPrimMesh& prim);
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor);
  void updateFrame();
  void resetFrame();
  void updateTopLevelAS();

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                        m_rtBuilder;
  nvvk::DescriptorSetBindings                       m_rtDescSetLayoutBind;
  VkDescriptorPool                                  m_rtDescPool;
  VkDescriptorSetLayout                             m_rtDescSetLayout;
  VkDescriptorSet                                   m_rtDescSet;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;
  nvvk::SBTWrapper                                  m_sbtWrapper;
  std::vector<VkAccelerationStructureInstanceKHR>   m_tlas;
  VkBuildAccelerationStructureFlagsKHR               m_tlasFlag;
  PushConstantRay m_pcRay{};
};
