<template>
  <el-container class="app-root">
    <el-aside width="220px" class="app-aside">
      <div class="logo">RiseQuant <small>Pattern-Cluster</small></div>
      <el-menu :default-active="active" router class="menu">
        <el-menu-item index="/data"><el-icon><DataAnalysis /></el-icon>数据</el-menu-item>
        <el-menu-item index="/strategy"><el-icon><Setting /></el-icon>策略编写</el-menu-item>
        <el-menu-item index="/backtest"><el-icon><Money /></el-icon>回测</el-menu-item>
        <el-menu-item index="/optimize"><el-icon><MagicStick /></el-icon>优化</el-menu-item>
        <el-menu-item index="/bundles"><el-icon><Box /></el-icon>训练产物</el-menu-item>
        <el-menu-item index="/jobs"><el-icon><Clock /></el-icon>任务</el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="app-header">
        <span class="page-title">{{ pageTitle }}</span>
      </el-header>
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
const route = useRoute()
const active = computed(() => route.path)
const titles = {
  '/data': '数据获取 / 清洗 / Dry-run',
  '/strategy': '策略编写',
  '/backtest': '回测',
  '/optimize': '参数优化',
  '/bundles': '训练产物可视化',
  '/jobs': '任务列表',
}
const pageTitle = computed(() => titles[route.path] || '')
</script>

<style>
html, body, #app { height: 100%; margin: 0; }
.app-root { height: 100vh; }
.app-aside { background: #001529; color: #fff; }
.logo { color: #fff; padding: 16px; font-size: 18px; font-weight: 600; }
.logo small { display: block; color: #8c8c8c; font-size: 12px; font-weight: normal; }
.menu { border-right: none; background: transparent; }
.menu .el-menu-item { color: #c0c4cc; }
.menu .el-menu-item.is-active { background: #1890ff !important; color: #fff !important; }
.app-header { background: #fff; border-bottom: 1px solid #eee; display: flex; align-items: center; }
.page-title { font-size: 18px; font-weight: 500; }
.app-main { background: #f5f7fa; padding: 20px; }
</style>
