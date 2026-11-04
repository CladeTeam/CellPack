<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import StoryIcon from "./StoryIcon.vue";

const props = defineProps({
  chapters: Array,
  visible: Boolean,
  activeChapter: String,
  activeScene: String,
  progress: Array,
  pending: String,
});
const emit = defineEmits(["navigate"]);
const open = ref(false);
const nav = ref(null);
const toggle = ref(null);
const plain = (html) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();

function navigate(event, id) {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button > 0) return;
  event.preventDefault();
  emit("navigate", id);
  if (open.value) close(true);
}

async function close(restoreFocus = false) {
  open.value = false;
  if (restoreFocus) {
    await nextTick();
    toggle.value?.focus({ preventScroll: true });
  }
}

function outside(event) {
  if (!nav.value?.contains(event.target)) close();
}

watch(() => props.visible, (value) => { if (!value) close(); });
onMounted(() => document.addEventListener("pointerdown", outside));
onBeforeUnmount(() => document.removeEventListener("pointerdown", outside));
</script>

<template>
  <nav
    v-show="visible"
    ref="nav"
    class="story-progress vp-raw"
    :class="{ 'is-open': open, 'is-travelling': pending }"
    aria-label="Story progress"
    @keydown.esc.stop.prevent="close(true)"
    @focusout="event => { if (!event.currentTarget.contains(event.relatedTarget)) close(); }"
  >
    <div class="story-progress-spine">
      <span class="story-progress-caption" aria-hidden="true">IN THIS STORY</span>
      <a class="story-progress-home" href="#home" aria-label="Home page: CellPack overview" @click="navigate($event, 'home')">
        <span class="story-progress-icon"><StoryIcon name="home" /></span>
        <span class="story-progress-label"><b>Home page</b><span>CellPack overview</span></span>
        <span v-if="!open" class="story-progress-tooltip" aria-hidden="true">Home page</span>
      </a>
      <ol class="story-progress-stops">
        <li v-for="(chapter, index) in chapters" :key="chapter.id" :class="{ 'is-current': activeChapter === chapter.id, 'is-complete': progress[index] >= 1, 'is-pending': pending === chapter.id }">
          <a
            :href="`#${chapter.id}`"
            :aria-label="`${chapter.label}: ${chapter.summary}`"
            :aria-current="activeChapter === chapter.id ? 'step' : undefined"
            @click="navigate($event, chapter.id)"
          >
            <span class="story-progress-icon"><StoryIcon :name="chapter.icon" /></span>
            <span class="story-progress-label"><b>{{ chapter.label }}</b><span>{{ chapter.summary }}</span></span>
            <span v-if="!open" class="story-progress-tooltip" aria-hidden="true">{{ chapter.label }}</span>
          </a>
          <span v-if="index < chapters.length - 1" class="story-progress-track" aria-hidden="true"><i :style="{ transform: `scaleY(${progress[index]})` }" /></span>
        </li>
      </ol>
      <button ref="toggle" class="story-progress-toggle" type="button" :aria-expanded="open" aria-controls="story-contents" :aria-label="open ? 'Close story contents' : 'Open story contents'" @click="open = !open">
        <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M8 5h9M8 10h9M8 15h9"/><g fill="currentColor" stroke="none"><circle cx="4" cy="5" r="1"/><circle cx="4" cy="10" r="1"/><circle cx="4" cy="15" r="1"/></g></svg><span>All scenes</span>
      </button>
    </div>
    <Transition name="story-contents">
      <div v-if="open" id="story-contents" class="story-contents-panel">
        <div class="story-contents-heading"><span>JUMP TO A TOPIC</span><span>{{ chapters.length }} sections</span></div>
        <ol class="story-contents-list">
          <li v-for="chapter in chapters" :key="chapter.id">
            <a class="story-contents-chapter" :class="{ 'is-current': activeChapter === chapter.id }" :href="`#${chapter.id}`" :aria-current="activeChapter === chapter.id && !activeScene ? 'location' : undefined" @click="navigate($event, chapter.id)"><StoryIcon :name="chapter.icon" /><span>{{ chapter.label }}</span></a>
            <ol>
              <li v-for="scene in chapter.scenes" :key="scene.id">
                <a :href="`#${scene.id}`" class="story-contents-scene" :class="{ 'is-current': activeScene === scene.id, 'is-pending': pending === scene.id }" :aria-current="activeScene === scene.id ? 'location' : undefined" @click="navigate($event, scene.id)"><span class="story-contents-dot" aria-hidden="true"/><span>{{ scene.label || plain(scene.beats[0].title) }}</span></a>
              </li>
            </ol>
          </li>
        </ol>
        <p>Jump to a scene. Scroll to explore it.</p>
      </div>
    </Transition>
  </nav>
</template>
