import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/data' },
  { path: '/data', name: 'data', component: () => import('./views/Data.vue') },
  { path: '/strategy', name: 'strategy', component: () => import('./views/Strategy.vue') },
  { path: '/backtest', name: 'backtest', component: () => import('./views/Backtest.vue') },
  { path: '/optimize', name: 'optimize', component: () => import('./views/Optimize.vue') },
  { path: '/bundles', name: 'bundles', component: () => import('./views/Bundles.vue') },
  { path: '/jobs', name: 'jobs', component: () => import('./views/Jobs.vue') },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
