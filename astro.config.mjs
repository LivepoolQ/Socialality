/*
 * @Author: Ziqian Zou
 * @Date: 2026-09-24 09:01:29
 * @LastEditors: Ziqian Zou
 * @LastEditTime: 2026-09-24 09:25:09
 * @Description: file content
 * @Github: https://github.com/LivepoolQ
 * Copyright 2026 Ziqian Zou, All Rights Reserved.
 */
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://LivepoolQ.github.io',
  base: '/Socialality',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },
  integrations: [mdx()],
});

