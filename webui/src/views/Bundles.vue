<template>
  <div>
    <el-card>
      <template #header>
        <b>训练产物 (bundles)</b>
        <el-button size="small" @click="loadBundles" style="float: right">刷新</el-button>
      </template>
      <el-table :data="bundles" highlight-current-row @row-click="selectBundle">
        <el-table-column prop="name" label="名称" min-width="200" />
        <el-table-column label="cluster 数" width="100">
          <template #default="{ row }">{{ row.meta && row.meta.n_clusters }}</template>
        </el-table-column>
        <el-table-column label="方案" width="180">
          <template #default="{ row }">
            <el-tag v-if="row.has_per_cluster" size="small">A</el-tag>
            <el-tag v-if="row.has_single_with_cf" type="success" size="small" style="margin-left: 4px">B</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="训练区间" min-width="200">
          <template #default="{ row }">
            {{ row.meta && row.meta.training_data_range && row.meta.training_data_range.join(' ~ ') }}
          </template>
        </el-table-column>
        <el-table-column label="silhouette" width="100">
          <template #default="{ row }">
            {{ row.meta && row.meta.training_config && row.meta.training_config.silhouette
                && row.meta.training_config.silhouette.toFixed(3) }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card v-if="detail" style="margin-top: 16px;">
      <template #header><b>{{ detail.bundle_dir }}</b></template>

      <el-divider content-position="left">cluster 统计</el-divider>
      <el-table :data="detail.cluster_stats || []" size="small">
        <el-table-column prop="cluster_id" label="cluster" />
        <el-table-column prop="n_samples" label="样本数" />
        <el-table-column prop="n_positive" label="正样本" />
        <el-table-column prop="n_golden_event_samples" label="来自金标准事件" />
      </el-table>

      <el-divider content-position="left">per-cluster 模型指标 (方案 A)</el-divider>
      <el-table v-if="detail.per_cluster" :data="Object.values(detail.per_cluster)" size="small">
        <el-table-column prop="cluster_id" label="cluster" width="80" />
        <el-table-column label="usable" width="80">
          <template #default="{ row }">
            <el-tag :type="row.usable ? 'success' : 'danger'" size="small">{{ row.usable ? 'yes' : 'no' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="AUC" width="100">
          <template #default="{ row }">{{ row.metrics && row.metrics.auc && row.metrics.auc.toFixed(3) }}</template>
        </el-table-column>
        <el-table-column label="precision@top10%" width="160">
          <template #default="{ row }">{{ row.metrics && row.metrics.precision_top10pct && row.metrics.precision_top10pct.toFixed(3) }}</template>
        </el-table-column>
        <el-table-column label="brier" width="100">
          <template #default="{ row }">{{ row.metrics && row.metrics.brier_score && row.metrics.brier_score.toFixed(4) }}</template>
        </el-table-column>
        <el-table-column label="signal@0.6" width="120">
          <template #default="{ row }">{{ row.metrics && row.metrics.signal_ratio_at_06 && (row.metrics.signal_ratio_at_06 * 100).toFixed(2) + '%' }}</template>
        </el-table-column>
        <el-table-column label="top-5 特征">
          <template #default="{ row }">
            <span v-for="f in (row.feature_importance || []).slice(0, 5)" :key="f.feature" style="margin-right: 8px;">
              {{ f.feature }} ({{ f.importance }})
            </span>
          </template>
        </el-table-column>
      </el-table>

      <el-divider content-position="left">可视化图</el-divider>
      <el-row :gutter="16">
        <el-col v-for="v in vizFiles" :key="v" :span="12" style="margin-bottom: 16px">
          <el-card shadow="hover">
            <div style="font-weight: 500; margin-bottom: 8px">{{ v }}</div>
            <img v-if="vizImages[v]" :src="vizImages[v]" style="width: 100%; max-height: 360px; object-fit: contain"
                  @click="onImageClick(v)" />
            <el-button v-else size="small" @click="loadViz(v)">加载</el-button>
          </el-card>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { bundleApi } from '../api.js'

const bundles = ref([])
const detail = ref(null)
const vizFiles = ref([])
const vizImages = ref({})

async function loadBundles() {
  const { data } = await bundleApi.list('./models/pattern_cluster')
  bundles.value = data
}

async function selectBundle(row) {
  detail.value = null
  vizImages.value = {}
  const { data } = await bundleApi.detail(row.bundle_dir)
  detail.value = data
  const { data: l } = await bundleApi.vizList(row.bundle_dir)
  vizFiles.value = (l.files || []).filter(n => n.toLowerCase().endsWith('.png'))
  // 自动加载前几张
  for (const v of vizFiles.value.slice(0, 4)) {
    await loadViz(v)
  }
}

async function loadViz(name) {
  if (!detail.value) return
  const { data } = await bundleApi.viz(detail.value.bundle_dir, name)
  vizImages.value[name] = `data:image/png;base64,${data.base64}`
}

function onImageClick(name) {
  const url = vizImages.value[name]
  if (url) window.open(url, '_blank')
}

onMounted(loadBundles)
</script>
