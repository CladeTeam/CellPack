<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import { withBase } from "vitepress";
import { canonicalLink } from "../routes.mjs";
import HeroScene from "./components/HeroScene.vue";
import SiteFooter from "./components/SiteFooter.vue";
import ScrollScene from "./home/ScrollScene.vue";
import ChapterArt from "./home/ChapterArt.vue";
import StoryProgress from "./home/StoryProgress.vue";
import StoryIcon from "./home/StoryIcon.vue";
import { useStoryNavigation } from "./home/useStoryNavigation.js";
import { chapters } from "./home/story.js";
import "./home.css";
import "./narrative.css";
import "./home/scenes.css";
import "./home/story-progress.css";

const link = (path) => withBase(canonicalLink(path));
const story = ref(null);
const { phase, visible, activeChapter, activeScene, progress, pending, announcement, jump } = useStoryNavigation(story, chapters);
let observer;
onMounted(() => {
  // Chapter cards fade in once when they reach the viewport.
  observer = new IntersectionObserver(
    (entries) => entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in-view");
      observer.unobserve(e.target);
    }),
    { threshold: 0.35 },
  );
  document.querySelectorAll(".chapter-text").forEach((el) => observer.observe(el));
});
onBeforeUnmount(() => observer?.disconnect());
</script>

<template>
  <main class="cp-home cp-story-home" :data-story-transition="phase">
    <header id="home" class="story-hero story-wrap home-hero">
      <div class="hero-copy">
        <p class="story-kicker">STORAGE + LOADING FOR SINGLE-CELL TRAINING</p>
        <h1 class="home-title" tabindex="-1">CellPack</h1>
        <p class="home-tagline">
          A storage format for billion-cell corpora that is easy to use,
          integrate and share — and a loading pipeline that keeps training fed,
          epoch after epoch.
        </p>
        <div class="home-actions">
          <a class="home-button primary" :href="link('/quickstart.html')">Get started <span aria-hidden="true">→</span></a>
          <a class="home-button" :href="link('/python/docs/')">Documentation</a>
          <a class="home-link" :href="link('/report.html')">Technical report ↗</a>
        </div>
        <dl class="home-stats">
          <div><dt>5.89×</dt><dd>compression ratio, one cell per decode</dd></div>
          <div><dt>9.2×</dt><dd>smaller than uncompressed H5AD, end to end</dd></div>
          <div><dt>3.6M <span class="home-stat-unit">cells/s</span></dt><dd>sustained on a billion-cell corpus</dd></div>
        </dl>
      </div>
      <HeroScene />
      <a class="home-scroll" href="#story" aria-label="Explore how we do it"
        ><span>How we do it</span><i aria-hidden="true" /></a>
    </header>

    <StoryProgress :chapters="chapters" :visible="visible" :active-chapter="activeChapter" :active-scene="activeScene" :progress="progress" :pending="pending" @navigate="jump" />
    <p class="story-navigation-status" role="status" aria-live="polite">{{ announcement }}</p>
    <div id="story" ref="story">
      <template v-for="(chapter, chapterIndex) in chapters" :key="chapter.id">
        <section :id="chapter.id" class="chapter-card story-wrap">
          <div class="chapter-text">
            <p class="chapter-label"><StoryIcon :name="chapter.icon" /><span>{{ chapter.label }}</span></p>
            <h2 v-html="chapter.title" />
            <p class="chapter-lede">{{ chapter.lede }}</p>
          </div>
          <ChapterArt :seed="(chapterIndex + 1) * 7 + chapter.id.length" />
        </section>
        <ScrollScene
          v-for="s in chapter.scenes"
          :id="s.id"
          :key="s.id"
          :scene="s.scene"
          :beats="s.beats"
          :length="s.length"
          :source="s.source"
          :layout="s.layout"
        />
      </template>
    </div>

    <section class="scene-cta story-wrap">
      <p class="story-kicker">START HERE</p>
      <h2>Bring CellPack <br />to your training data.</h2>
      <div class="cta-grid">
        <a :href="link('/quickstart.html')"
          ><b>Quickstart</b><span>Convert a matrix and load your first batches.</span><i>↗</i></a
        ><a :href="link('/overview.html')"
          ><b>Documentation</b><span>Storage, access sequences, cache planning and the format.</span><i>↗</i></a
        ><a :href="link('/api.html')"
          ><b>API reference</b><span>Every public type, with runnable examples.</span><i>↗</i></a
        ><a :href="link('/report.html')"
          ><b>Technical report</b><span>Design, experiments and the data behind this page.</span><i>↗</i></a
        >
      </div>
      <p class="scene-footnote">
        Benchmark values on this page are frozen results from the CellPack
        technical report, not new measurements of the current codebase. Format
        step sizes were measured on a sample of CELLxGENE chunks; loader
        diagrams are explanatory.
        <a :href="link('/research/home/sources.json')">Data provenance ↗</a>
      </p>
    </section>
    <SiteFooter />
  </main>
</template>
