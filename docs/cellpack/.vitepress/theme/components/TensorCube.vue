<script setup>
import { computed } from "vue";
const props = defineProps({ seed: { type: Number, default: 0 } });
const shape = { width: 4, depth: 5, height: 3 };
const project = (x, y, z) => [61 + (x - y) * 10, 45 + (x + y) * 5.8 - z * 11.6];
function polygon(corners) {
  const points = corners.map(([x, y, z]) => project(x, y, z));
  const centre = points.reduce(
    ([x, y], point) => [x + point[0] / 4, y + point[1] / 4],
    [0, 0],
  );
  return points
    .map(
      ([x, y]) =>
        `${centre[0] + (x - centre[0]) * 0.94},${centre[1] + (y - centre[1]) * 0.94}`,
    )
    .join(" ");
}
const tiles = computed(() => {
  const result = [];
  for (const face of ["left", "right", "top"]) {
    const columns = face === "right" ? shape.depth : shape.width;
    const rows = face === "top" ? shape.depth : shape.height;
    for (let a = 0; a < columns; a++)
      for (let b = 0; b < rows; b++) {
        const corners =
          face === "top"
            ? [
                [a, b, shape.height],
                [a + 1, b, shape.height],
                [a + 1, b + 1, shape.height],
                [a, b + 1, shape.height],
              ]
            : face === "left"
              ? [
                  [a, shape.depth, b],
                  [a + 1, shape.depth, b],
                  [a + 1, shape.depth, b + 1],
                  [a, shape.depth, b + 1],
                ]
              : [
                  [shape.width, a, b],
                  [shape.width, a + 1, b],
                  [shape.width, a + 1, b + 1],
                  [shape.width, a, b + 1],
                ];
        const value = a * 7 + b * 3 + props.seed * 5 + (face === "top" ? 3 : 0);
        result.push({
          key: `${face}-${a}-${b}`,
          face,
          points: polygon(corners),
          filled: value % 7 < 3,
          warm: value % 13 === 2,
        });
      }
  }
  return result;
});
</script>

<template>
  <svg
    class="cp-tensor-cube"
    viewBox="0 0 112 108"
    aria-hidden="true"
    style="display: block; width: 100%; height: 100%"
  >
    <polygon
      v-for="tile in tiles"
      :key="tile.key"
      :points="tile.points"
      class="cp-tensor-tile"
      :class="[tile.face, { filled: tile.filled, warm: tile.warm }]"
    />
  </svg>
</template>

<style scoped>
.cp-tensor-tile {
  stroke: var(--vp-c-bg);
  stroke-width: 0.65;
  stroke-linejoin: round;
}
.left {
  fill: color-mix(in srgb, var(--cp-teal-soft) 70%, var(--cp-teal));
}
.right {
  fill: color-mix(in srgb, var(--cp-teal-soft) 90%, var(--cp-teal));
}
.top {
  fill: var(--cp-teal-soft);
}
.left.filled {
  fill: color-mix(in srgb, var(--cp-teal) 80%, var(--vp-c-text-1));
}
.right.filled {
  fill: var(--cp-teal);
}
.top.filled {
  fill: color-mix(in srgb, var(--cp-teal) 65%, var(--vp-c-bg));
}
.cp-tensor-tile.warm {
  fill: var(--cp-accent);
}
</style>
