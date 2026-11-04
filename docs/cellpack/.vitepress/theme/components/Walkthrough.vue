<script setup>
import { computed } from "vue";
import { useDiagramPlayback } from "./useDiagramPlayback.js";
const props = defineProps({
  steps: { type: Array, required: true },
  label: String,
  stepDuration: { type: Number, default: 3000 },
});
const total = computed(() => props.steps.length * props.stepDuration);
const { root, elapsed, playing, ended, state, toggle } =
  useDiagramPlayback(total);
const stage = computed(() =>
  Math.min(
    props.steps.length - 1,
    Math.floor(elapsed.value / props.stepDuration),
  ),
);
const progress = computed(() =>
  ended.value ? 1 : (elapsed.value % props.stepDuration) / props.stepDuration,
);
const action = computed(
  () =>
    `${ended.value ? "Replay" : playing.value ? "Pause" : "Continue"} ${props.label}`,
);
</script>
<template>
  <figure class="walkthrough" :aria-label="label" :data-state="state">
    <div class="walk-top">
      <span>{{ label }}</span
      ><span class="walk-interaction-hint" aria-hidden="true"
        >Click diagram</span
      >
    </div>
    <div
      ref="root"
      class="walk-canvas"
      role="button"
      tabindex="0"
      :aria-label="action"
      :aria-pressed="playing"
      @click="toggle"
      @keydown.enter.prevent="toggle"
      @keydown.space.prevent="toggle"
    >
      <slot
        :stage="stage"
        :progress="progress"
        :playing="playing"
        :elapsed="elapsed"
      />
    </div>
    <div v-if="$slots.controls" class="walk-extra-controls">
      <slot name="controls" />
    </div>
    <figcaption>
      <span class="step-number">{{ String(stage + 1).padStart(2, "0") }}</span
      ><span>{{ steps[stage] }}</span>
    </figcaption>
    <div class="walk-progress" aria-hidden="true">
      <span v-for="(_, i) in steps" :key="i"
        ><i
          :style="{
            transform: `scaleX(${i < stage ? 1 : i === stage ? progress : 0})`,
          }"
      /></span>
    </div>
  </figure>
</template>
