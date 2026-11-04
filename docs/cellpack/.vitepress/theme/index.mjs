import { defineAsyncComponent } from "vue";
import { withBase } from "vitepress";
import DefaultTheme from "vitepress/theme-without-fonts";
import SiteLayout from './site/SiteLayout.vue';
import { canonicalLink } from '../routes.mjs';
import "@fontsource-variable/dm-sans";
import "@fontsource-variable/space-grotesk";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "./logo.css";
import "./style.css";
import "./api/api.css";
import './site/site.css';

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router, siteData }) {
    app.component('DocDiagram', defineAsyncComponent(() => import('./components/DocDiagram.vue')));
    app.component('DocChoices', defineAsyncComponent(() => import('./components/DocChoices.vue')));
    app.component("ApiExplorer", defineAsyncComponent(() => import("./api/ApiExplorer.vue")));
    app.component("ApiExample", defineAsyncComponent(() => import("./api/ApiExample.vue")));
    app.component("ApiToolbar", defineAsyncComponent(() => import("./api/ApiToolbar.vue")));
    app.component("ApiMemberIndex", defineAsyncComponent(() => import("./api/ApiMemberIndex.vue")));
    app.component('SubsiteHome', defineAsyncComponent(() => import('./site/SubsiteHome.vue')));
    const previous = router.onBeforeRouteChange;
    router.onBeforeRouteChange = async target => {
      const base = siteData.value.base;
      const url = new URL(target, 'https://docs.invalid');
      const local = '/' + url.pathname.slice(base.length);
      const canonical = canonicalLink(local);
      if (canonical !== local && url.pathname.startsWith(base)) {
        await router.go(withBase(canonical) + url.search + url.hash);
        return false;
      }
      return previous?.(target);
    };
  },
  Layout: SiteLayout,
};
