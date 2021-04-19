// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision highp float;
in vec4 vColor;
in vec3 vPosition;

float rand(vec2 co) {
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

vec3 getNormal() {
  vec3 dPositiondx = dFdx(vPosition);
  vec3 dPositiondy = dFdy(vPosition);
  return normalize(cross(dPositiondx, dPositiondy));
}

float fresnel(float f0, vec3 n, vec3 l){
  return f0 + (1.0-f0) * pow(1.0- dot(n, l), 5.0);
}

float diffuseEnergyRatio(float f0, vec3 n, vec3 l) {
  return 1.0 - fresnel(f0, n, l);
}

// Some of the code below is adapted from: https://github.com/IonutCava/Divide-Framework  (MIT License)
//Beckmann
float distribution(vec3 n, vec3 h, float roughness){
  float m_Sq= roughness * roughness;
  float NdotH_Sq= max(dot(n, h), 0.0);
  NdotH_Sq= NdotH_Sq * NdotH_Sq;
  return exp( (NdotH_Sq - 1.0)/(m_Sq*NdotH_Sq) )/ (3.14159265 * m_Sq * NdotH_Sq * NdotH_Sq) ;
}

// Schlick
float geometry(vec3 n, vec3 h, vec3 v, vec3 l, float roughness){
  float NdotH= dot(n, h);
  float NdotL= dot(n, l);
  float NdotV= dot(n, v);
  float VdotH= dot(v, h);
  float NdotL_clamped= max(NdotL, 0.0);
  float NdotV_clamped= max(NdotV, 0.0);
  return min( min( 2.0 * NdotH * NdotV_clamped / VdotH, 2.0 * NdotH * NdotL_clamped / VdotH), 1.0);
}

vec3 getBrdfTerm( float u_roughness,
                  vec3 u_diffuseColor,
                  vec3 u_lightColor,
                  vec3 u_lightDir,
                  float refractiveIndex) {
  float u_fresnel0 = (1.0 - refractiveIndex)/(1.0 + refractiveIndex);
  u_fresnel0 = u_fresnel0 * u_fresnel0;

  vec3 normal =  getNormal();
  vec3 view   = -normalize(vPosition - cameraPosition); //vec3(0, 1.9, 3.7));
  vec3 halfVec=  normalize(u_lightDir + view);
  float NdotL= dot(normal, u_lightDir);
  float NdotV= dot(normal, view);
  float NdotL_clamped= max(NdotL, 0.0);
  float NdotV_clamped= max(NdotV, 0.0);

  float brdf_spec= fresnel(u_fresnel0, halfVec, u_lightDir) *
    geometry(normal, halfVec, view, u_lightDir, u_roughness) *
    distribution(normal, halfVec, u_roughness) / (4.0 * NdotL_clamped * NdotV_clamped);
  vec3 color_spec= NdotL_clamped * brdf_spec * u_lightColor;
  vec3 color_diff= NdotL_clamped * diffuseEnergyRatio(u_fresnel0, normal, u_lightDir) * u_diffuseColor * u_lightColor;
  return color_diff + color_spec;
}

vec4 mix_diffuse(vec3 lightPos, vec3 normal, vec4 color, vec4 light_color, float min_val, float max_val) {
  float diffuse = dot(normalize(lightPos - vPosition), normal);
  diffuse = max(min_val, diffuse);
  return mix(color, light_color, diffuse * max_val);
}

vec4 mix_specular(vec3 lightPos, vec3 normal, vec4 color, vec4 light_color, float min_val, float max_val) {
  vec3 halfVector = normalize(normalize(cameraPosition - vPosition) + normalize(lightPos - vPosition));
  float specular = dot(normal, halfVector);
  specular = max(min_val, specular);
  specular = pow(specular, 50.0);
  return mix(color, light_color, specular * max_val);
}

vec4 default_plastic2(vec4 color, vec4 dark, vec4 light) {
  vec3 sunPos = cameraPosition * 10.0;
  vec3 spotPos = cameraPosition * 8.0 + vec3(0.0, -3.0, -0.0);
  vec4 spotColor = vec4(1.0, 1.0, 0.0, 1.0);

  vec3 normal = getNormal();
  color = mix_diffuse(sunPos, normal, dark, color, 0.1, 1.0);
  color = mix_diffuse(spotPos, normal, color, spotColor, 0.0, 0.2);
  color = mix_specular(sunPos, normal, color, light, 0.0, 0.5);
  color = mix_specular(spotPos, normal, color, spotColor, 0.0, 0.5);
  return color;
}

bool isNan(float val) {
  return (val <= 0.0 || 0.0 <= val) ? false : true;
}

void main() {
  vec3 lightDir = normalize(vec3(-1.0, 1.0, 1.0));
  lightDir = normalize(cameraPosition);
  vec3 brdf = getBrdfTerm(0.24,
                          vec3(0.1/3.14, 1.0/3.14, 0.3/3.14),
                          vec3(1.0, 0.8, 0.8) * 3.0,
                          lightDir,
                          2.0);
  vec4 brdf_vec4 = vec4(
    max(0.0, min(brdf[0], 1.0)),
    max(0.0, min(brdf[1], 1.0)),
    max(0.0, min(brdf[2], 1.0)),
    1.0);

  vec4 plast2 = default_plastic2(
    vec4(0.2, 0.8, 0.67, 1.0),
    vec4(0.46, 0.72, 0.0, 1.0),
    vec4(1.0, 1.0, 1.0, 1.0));

  vec4 composite =  0.4 * brdf_vec4 + plast2 * 0.6;

  gl_FragColor = composite;
}
