<script setup>
import HeroLogo from "../HeroLogo.vue";
import TensorCube from "./TensorCube.vue";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
const props = defineProps({ paused: Boolean });
const scene = ref(null);
const visible = ref(false);
const hidden = ref(false);
const paused = computed(() => props.paused || hidden.value || !visible.value);
function visibility() { hidden.value = document.hidden; }
let observer;
onMounted(() => {
  visibility();
  document.addEventListener("visibilitychange", visibility);
  observer = new IntersectionObserver(
    ([entry]) => {
      visible.value = entry.isIntersecting;
    },
    { threshold: 0.12 },
  );
  observer.observe(scene.value);
});
onBeforeUnmount(() => {
  observer?.disconnect();
  document.removeEventListener("visibilitychange", visibility);
});
const tensors = [0, 1, 2];
</script>

<template>
  <div
    ref="scene"
    class="cp-hero-scene"
    :class="{ 'in-view': visible, 'motion-paused': paused }"
    aria-label="Sparse matrix records flow through CellPack into tensor batches on a conveyor feeding training"
    role="group"
  >
    <div class="cp-scene-topline">
      <span class="cp-micro">A LITTLE LESS DATA. A LOT MORE FLOW.</span>
    </div>
    <div class="cp-orbit cp-orbit-one" aria-hidden="true" />
    <div class="cp-orbit cp-orbit-two" aria-hidden="true" />
    <div class="cp-matrix-block" aria-hidden="true">
      <div class="cp-sparse-grid">
        <i
          v-for="n in 64"
          :key="n"
          :class="{
            filled: n % 7 === 0 || n % 11 === 0 || n === 26,
            warm: n % 11 === 0,
          }"
        />
      </div>
      <span class="cp-micro">SPARSE MATRIX</span>
    </div>
    <div class="cp-hero-connection cp-connection-in" aria-hidden="true">
      <i /><i /><i />
    </div>
    <div class="cp-hero-logo"><HeroLogo :paused="paused" /></div>
    <div class="cp-hero-connection cp-connection-out" aria-hidden="true">
      <i /><i /><i />
    </div>
    <div class="cp-tensor-conveyor" aria-hidden="true">
      <div class="cp-training-port">
        <span>TRAIN</span
        ><svg viewBox="0 0 16 16"><path d="M8 13V3M4 7l4-4 4 4" /></svg>
      </div>
      <div class="cp-tensor-window">
        <div class="cp-tensor-belt">
          <div v-for="copy in 2" :key="copy" class="cp-tensor-group">
            <div v-for="tensor in tensors" :key="tensor" class="cp-tensor">
              <TensorCube :seed="tensor" />
            </div>
          </div>
        </div>
      </div>
      <span class="cp-micro">TENSOR BATCHES</span>
      <span class="cp-tensor-shape">B × N × D</span>
    </div>
    <div class="cp-scene-bottomline">
      <span><i /> INDEPENDENT RECORDS</span
      ><span>REQUESTED CELLS <span aria-hidden="true">↗</span></span>
    </div>
  </div>
</template>
