import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'
import router from './router.js'

const app = createApp(App)
for (const [k, v] of Object.entries(ElementPlusIconsVue)) {
  app.component(k, v)
}
app.use(ElementPlus)
app.use(router)
app.mount('#app')
