import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

// data
export const dataApi = {
  fetch: (body) => api.post('/data/fetch', body),
  clean: (body) => api.post('/data/clean', body),
  dryrun: (body) => api.post('/data/dryrun', body),
}

// train / backtest
export const trainApi = {
  submit: (body) => api.post('/train', body),
}
export const backtestApi = {
  submit: (body) => api.post('/backtest', body),
}

// bundles
export const bundleApi = {
  list: (base) => api.get('/bundles', { params: { base } }),
  detail: (bundle_dir) => api.get('/bundles/detail', { params: { bundle_dir } }),
  vizList: (bundle_dir) => api.get('/bundles/visualization-list', { params: { bundle_dir } }),
  viz: (bundle_dir, name) =>
    api.get('/bundles/visualization', { params: { bundle_dir, name } }),
  backtests: (base) => api.get('/bundles/backtests', { params: { base } }),
  backtestDetail: (run_dir) =>
    api.get('/bundles/backtest-detail', { params: { run_dir } }),
}

// optimize
export const optimizeApi = {
  search: (body) => api.post('/optimize/search', body),
  sensitivity: (body) => api.post('/optimize/sensitivity', body),
}

// jobs
export const jobApi = {
  list: (kind) => api.get('/jobs', { params: { kind } }),
  get: (id) => api.get(`/jobs/${id}`),
}

export default api
