<template>
  <el-card v-if="job" class="watcher">
    <template #header>
      <span>任务 #{{ job.id }} — {{ job.kind }}</span>
      <el-tag :type="stateType" style="margin-left: 8px">{{ job.state }}</el-tag>
    </template>
    <div v-if="job.error" style="color: #c00; margin-bottom: 8px;">
      <b>{{ job.error.type }}:</b> {{ job.error.msg }}
    </div>
    <div v-if="job.log_path" class="log-path">
      log: {{ job.log_path }}
    </div>
    <el-collapse v-if="job.result">
      <el-collapse-item title="result">
        <pre>{{ JSON.stringify(job.result, null, 2) }}</pre>
      </el-collapse-item>
    </el-collapse>
    <el-collapse v-if="job.logs && job.logs.length">
      <el-collapse-item :title="`logs (${job.logs.length})`">
        <pre class="logs">{{ job.logs.slice(-100).join('\n') }}</pre>
      </el-collapse-item>
    </el-collapse>
  </el-card>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'
import { jobApi } from '../api.js'

const props = defineProps({ jobId: String })
const job = ref(null)
let timer = null

const stateType = computed(() => {
  if (!job.value) return ''
  return { pending: 'info', running: 'warning', done: 'success', failed: 'danger' }[job.value.state] || ''
})

async function poll() {
  if (!props.jobId) return
  try {
    const { data } = await jobApi.get(props.jobId)
    job.value = data
    if (['done', 'failed'].includes(data.state)) {
      clearInterval(timer); timer = null
    }
  } catch (e) {
    console.error(e)
  }
}

watch(() => props.jobId, () => {
  if (timer) { clearInterval(timer); timer = null }
  job.value = null
  if (props.jobId) {
    poll()
    timer = setInterval(poll, 2000)
  }
}, { immediate: true })

onUnmounted(() => { if (timer) clearInterval(timer) })
</script>

<style scoped>
.watcher { margin-top: 16px; }
.log-path { color: #666; font-size: 12px; margin-bottom: 8px; word-break: break-all; }
.logs { background: #1e1e1e; color: #e0e0e0; padding: 8px; max-height: 300px; overflow: auto; font-size: 12px; }
pre { margin: 0; }
</style>
