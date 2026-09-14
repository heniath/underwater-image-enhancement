// =========================================================
// Kaggle Control Hub - Modern Vanilla JavaScript Engine
// =========================================================

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    activeProject: 'pdf_ocr_benchmark',
    accounts: [],
    projects: [],
    jobs: [],
    selectedLogFile: '',
    logSearchTerm: '',
    autoRefreshInterval: null,
    chartErrorsInstance: null,
    chartSpeedInstance: null,
  };

  // DOM Elements
  const tabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const projectSelector = document.getElementById('projectSelector');
  const accountsGrid = document.getElementById('accountsGrid');
  const jobsTableBody = document.getElementById('jobsTableBody');
  const threadCountBadge = document.getElementById('threadCountBadge');
  const totalAccountsVal = document.getElementById('totalAccountsVal');
  const totalActiveSlotsVal = document.getElementById('totalActiveSlotsVal');
  const totalJobsVal = document.getElementById('totalJobsVal');

  // Quick Action Buttons
  const btnQuickRunAll = document.getElementById('btnQuickRunAll');
  const btnQuickRefresh = document.getElementById('btnQuickRefresh');
  const btnQuickDownload = document.getElementById('btnQuickDownload');
  const btnScanQuotas = document.getElementById('btnScanQuotas');

  // Modal Elements
  const modalAddJob = document.getElementById('modalAddJob');
  const btnOpenAddJobModal = document.getElementById('btnOpenAddJobModal');
  const btnCloseAddJobModal = document.getElementById('btnCloseAddJobModal');
  const btnCancelAddJob = document.getElementById('btnCancelAddJob');
  const formAddJob = document.getElementById('formAddJob');
  const jobFormAccount = document.getElementById('jobFormAccount');
  const jobFormType = document.getElementById('jobFormType');
  const finetuneOptions = document.getElementById('finetuneOptions');

  // Chat Elements
  const chatMessages = document.getElementById('chatMessages');
  const chatInput = document.getElementById('chatInput');
  const btnSendChat = document.getElementById('btnSendChat');
  const promptPills = document.querySelectorAll('.prompt-pill');

  // Log Elements
  const logFileSelect = document.getElementById('logFileSelect');
  const logTailLines = document.getElementById('logTailLines');
  const logSearchInput = document.getElementById('logSearchInput');
  const logOutputContainer = document.getElementById('logOutputContainer');
  const currentLogName = document.getElementById('currentLogName');
  const btnRefreshLog = document.getElementById('btnRefreshLog');

  // Settings & Accounts Elements
  const formAddAccount = document.getElementById('formAddAccount');
  const tokenAccountSelect = document.getElementById('tokenAccountSelect');
  const btnUploadKaggleJson = document.getElementById('btnUploadKaggleJson');
  const fileKaggleJson = document.getElementById('fileKaggleJson');
  const inputApiKey = document.getElementById('inputApiKey');
  const btnSaveApiKey = document.getElementById('btnSaveApiKey');
  const btnValidateToken = document.getElementById('btnValidateToken');
  const tokenValidationResult = document.getElementById('tokenValidationResult');
  const btnGitStatus = document.getElementById('btnGitStatus');
  const btnGitPush = document.getElementById('btnGitPush');
  const gitStatusOutput = document.getElementById('gitStatusOutput');

  // ---------------------------------------------------------
  // 1. Tab Navigation Routing
  // ---------------------------------------------------------
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      tabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');

      if (targetId === 'tab-analytics') loadAnalytics();
      if (targetId === 'tab-logs') loadLogs();
      if (targetId === 'tab-threads') loadJobsStatus();
    });
  });

  // ---------------------------------------------------------
  // 2. Data Loading & Initialization
  // ---------------------------------------------------------
  async function init() {
    await loadProjects();
    await loadAccounts();
    await loadJobsStatus();
    loadAnalytics();
    setupEventListeners();
  }

  async function loadProjects() {
    try {
      const res = await fetch('/api/projects');
      const data = await res.json();
      state.projects = data.projects || [];
      projectSelector.innerHTML = '';
      state.projects.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name;
        projectSelector.appendChild(opt);
      });
      if (state.projects.length > 0) {
        state.activeProject = state.projects[0].name;
      }
    } catch (err) {
      showToast('Failed to load projects: ' + err.message, 'error');
    }
  }

  async function loadAccounts() {
    try {
      const res = await fetch('/api/accounts');
      const data = await res.json();
      state.accounts = data.accounts || [];
      renderAccountsGrid();
      updateAccountsDropdowns();
      updateDashboardStats();
    } catch (err) {
      showToast('Failed to load accounts: ' + err.message, 'error');
    }
  }

  function renderAccountsGrid() {
    accountsGrid.innerHTML = '';
    if (state.accounts.length === 0) {
      accountsGrid.innerHTML = '<div class="card" style="grid-column: 1/-1;">No accounts found. Use the Accounts & Git tab to add one.</div>';
      return;
    }

    state.accounts.forEach(acc => {
      const card = document.createElement('div');
      card.className = 'account-card';
      const pct = Math.min(100, (acc.gpu_hours_used / acc.gpu_hours_total) * 100);
      const badgeClass = acc.token_valid ? 'badge-active' : 'badge-error';
      const badgeText = acc.token_valid ? 'Token Valid' : 'Token Invalid / Missing';

      let runningInfo = '';
      if (acc.running_kernels && acc.running_kernels.length > 0) {
        runningInfo = `<div style="font-size: 0.78rem; color: #38bdf8; margin-top: 8px;">Active: <code>${acc.running_kernels.join(', ')}</code></div>`;
      }

      card.innerHTML = `
        <div class="account-card-header">
          <span class="account-card-title">${acc.id}</span>
          <span class="account-badge ${badgeClass}">${badgeText}</span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
          Username: <code style="color: #38bdf8;">${acc.username || 'N/A'}</code>
        </div>
        <div class="account-metric-row">
          <span>Active Batch GPU Slots:</span>
          <b>${acc.active_sessions} / ${acc.max_sessions} slots</b>
        </div>
        <div class="account-metric-row">
          <span>Remaining GPU Quota:</span>
          <b>${acc.gpu_hours_remaining.toFixed(1)}h / ${acc.gpu_hours_total.toFixed(0)}h</b>
        </div>
        <div class="progress-bar-wrapper">
          <div class="progress-bar-fill" style="width: ${pct}%;"></div>
        </div>
        ${runningInfo}
      `;
      accountsGrid.appendChild(card);
    });
  }

  function updateAccountsDropdowns() {
    jobFormAccount.innerHTML = '';
    tokenAccountSelect.innerHTML = '';
    state.accounts.forEach(acc => {
      const opt1 = document.createElement('option');
      opt1.value = acc.id;
      opt1.textContent = `${acc.id} (${acc.username})`;
      jobFormAccount.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = acc.id;
      opt2.textContent = `${acc.id} (${acc.username})`;
      tokenAccountSelect.appendChild(opt2);
    });
  }

  function updateDashboardStats() {
    totalAccountsVal.textContent = state.accounts.length;
    let activeSlots = 0;
    let maxSlots = state.accounts.length * 2;
    state.accounts.forEach(a => activeSlots += (a.active_sessions || 0));
    totalActiveSlotsVal.textContent = `${activeSlots} / ${maxSlots}`;
    totalJobsVal.textContent = state.jobs.length;
    threadCountBadge.textContent = state.jobs.length;
  }

  // ---------------------------------------------------------
  // 3. Multi-Thread Jobs Matrix
  // ---------------------------------------------------------
  async function loadJobsStatus() {
    if (!state.activeProject) return;
    try {
      const res = await fetch(`/api/projects/${state.activeProject}/jobs_status`);
      const data = await res.json();
      state.jobs = data.jobs || [];
      renderJobsTable();
      updateDashboardStats();
    } catch (err) {
      showToast('Failed to refresh thread status: ' + err.message, 'error');
    }
  }

  function renderJobsTable() {
    jobsTableBody.innerHTML = '';
    if (state.jobs.length === 0) {
      jobsTableBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No threads configured for this project.</td></tr>';
      return;
    }

    state.jobs.forEach(job => {
      const tr = document.createElement('tr');
      let statusBadge = '<span class="account-badge">IDLE</span>';
      if (job.status === 'COMPLETE') statusBadge = '<span class="account-badge badge-complete">COMPLETE</span>';
      else if (job.status === 'RUNNING') statusBadge = '<span class="account-badge badge-running">RUNNING</span>';
      else if (job.status === 'QUEUED') statusBadge = '<span class="account-badge badge-queued">QUEUED</span>';
      else if (job.status === 'ERROR') statusBadge = '<span class="account-badge badge-error">ERROR</span>';

      tr.innerHTML = `
        <td><strong>${job.name}</strong></td>
        <td><code>${job.account_id}</code> (${job.username})</td>
        <td><code>${job.kernel_id}</code></td>
        <td>
          <span style="font-size: 0.78rem; text-transform: uppercase; color: #94a3b8;">${job.job_type}</span><br>
          <code>${(job.engines || []).join(', ')}</code>
        </td>
        <td><span style="font-size: 0.8rem; color: #38bdf8;">${job.machine_shape}</span></td>
        <td>${statusBadge}</td>
        <td>
          <div class="table-actions">
            <button class="btn btn-primary btn-sm btn-run-job" data-job="${job.name}" title="Run thread">Run</button>
            <button class="btn btn-secondary btn-sm btn-dl-job" data-job="${job.name}" title="Download output">Download</button>
            <button class="btn btn-danger btn-sm btn-del-job" data-job="${job.name}" title="Delete thread">Delete</button>
          </div>
        </td>
      `;
      jobsTableBody.appendChild(tr);
    });

    // Action handlers
    document.querySelectorAll('.btn-run-job').forEach(btn => {
      btn.addEventListener('click', () => runSingleJob(btn.getAttribute('data-job')));
    });
    document.querySelectorAll('.btn-dl-job').forEach(btn => {
      btn.addEventListener('click', () => downloadSingleJob(btn.getAttribute('data-job')));
    });
    document.querySelectorAll('.btn-del-job').forEach(btn => {
      btn.addEventListener('click', () => deleteSingleJob(btn.getAttribute('data-job')));
    });
  }

  async function runSingleJob(jobName) {
    showToast(`Dispatching thread ${jobName}...`, 'info');
    try {
      const res = await fetch('/api/jobs/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_name: state.activeProject, job_names: [jobName] }),
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Thread ${jobName} dispatched successfully.`, 'success');
        loadJobsStatus();
      } else {
        showToast(`Dispatch failed: ${data.detail || 'Error'}`, 'error');
      }
    } catch (err) {
      showToast('Error sending command: ' + err.message, 'error');
    }
  }

  async function downloadSingleJob(jobName) {
    showToast(`Downloading outputs for thread ${jobName}...`, 'info');
    try {
      const res = await fetch('/api/jobs/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_name: state.activeProject, job_names: [jobName] }),
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Output artifacts downloaded for ${jobName}.`, 'success');
      }
    } catch (err) {
      showToast('Download error: ' + err.message, 'error');
    }
  }

  async function deleteSingleJob(jobName) {
    if (!confirm(`Are you sure you want to delete thread "${jobName}"?`)) return;
    try {
      const res = await fetch(`/api/jobs/${state.activeProject}/${jobName}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.success) {
        showToast(`Thread ${jobName} deleted.`, 'success');
        await loadProjects();
        await loadJobsStatus();
      }
    } catch (err) {
      showToast('Delete error: ' + err.message, 'error');
    }
  }

  // ---------------------------------------------------------
  // 4. Batch Operations
  // ---------------------------------------------------------
  btnQuickRunAll.addEventListener('click', async () => {
    showToast('Dispatching all threads in parallel...', 'info');
    try {
      const res = await fetch('/api/jobs/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_name: state.activeProject }),
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Dispatched ${data.count} thread(s) in parallel.`, 'success');
        loadJobsStatus();
      }
    } catch (err) {
      showToast('Parallel run error: ' + err.message, 'error');
    }
  });

  btnQuickRefresh.addEventListener('click', () => {
    loadAccounts();
    loadJobsStatus();
    loadAnalytics();
    showToast('System data refreshed.', 'success');
  });

  btnQuickDownload.addEventListener('click', async () => {
    showToast('Downloading all output artifacts from Kaggle...', 'info');
    try {
      const res = await fetch('/api/jobs/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_name: state.activeProject }),
      });
      const data = await res.json();
      if (data.success) {
        showToast('All artifacts downloaded.', 'success');
        loadAnalytics();
      }
    } catch (err) {
      showToast('Download error: ' + err.message, 'error');
    }
  });

  btnScanQuotas.addEventListener('click', () => {
    loadAccounts();
    showToast('Account quotas scanned and updated.', 'success');
  });

  // ---------------------------------------------------------
  // 5. Add Job Modal
  // ---------------------------------------------------------
  btnOpenAddJobModal.addEventListener('click', () => modalAddJob.classList.add('active'));
  btnCloseAddJobModal.addEventListener('click', () => modalAddJob.classList.remove('active'));
  btnCancelAddJob.addEventListener('click', () => modalAddJob.classList.remove('active'));

  jobFormType.addEventListener('change', () => {
    finetuneOptions.style.display = jobFormType.value === 'finetune' ? 'grid' : 'none';
  });

  formAddJob.addEventListener('submit', async (e) => {
    e.preventDefault();
    const checkedEngines = Array.from(document.querySelectorAll('input[name="engines"]:checked')).map(cb => cb.value);
    const newJob = {
      project_name: state.activeProject,
      name: document.getElementById('jobFormName').value.trim(),
      account_id: jobFormAccount.value,
      job_type: jobFormType.value,
      machine_shape: document.getElementById('jobFormGpu').value,
      engines: checkedEngines,
      install_files: [document.getElementById('jobFormReqs').value.trim()],
      limit: parseInt(document.getElementById('jobFormLimit').value, 10) || 20,
    };
    if (newJob.job_type === 'finetune') {
      newJob.finetune_model = document.getElementById('jobFormFtModel').value;
      newJob.epochs = parseInt(document.getElementById('jobFormEpochs').value, 10) || 10;
    }

    try {
      const res = await fetch('/api/jobs/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newJob),
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Thread ${newJob.name} created successfully.`, 'success');
        modalAddJob.classList.remove('active');
        formAddJob.reset();
        await loadProjects();
        await loadJobsStatus();
      } else {
        showToast(data.detail || 'Failed to create thread.', 'error');
      }
    } catch (err) {
      showToast('Request error: ' + err.message, 'error');
    }
  });

  // ---------------------------------------------------------
  // 6. AI Assistant & Chat Engine
  // ---------------------------------------------------------
  btnSendChat.addEventListener('click', () => sendChatMessage());
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });

  promptPills.forEach(pill => {
    pill.addEventListener('click', () => {
      chatInput.value = pill.getAttribute('data-prompt');
      sendChatMessage();
    });
  });

  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = '';

    appendChatMessage('user', text);
    const typingId = appendTypingIndicator();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, project_name: state.activeProject }),
      });
      const data = await res.json();
      removeTypingIndicator(typingId);
      appendChatMessage('assistant', data.response || 'Action completed.', data.actions);
    } catch (err) {
      removeTypingIndicator(typingId);
      appendChatMessage('assistant', 'Connection error with Assistant: ' + err.message);
    }
  }

  function appendChatMessage(role, content, actions = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let formatted = content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');

    let actionsHtml = '';
    if (actions && actions.length > 0) {
      actionsHtml = '<div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px;">' +
        actions.map(a => `<button class="btn btn-secondary btn-sm chat-action-btn" data-cmd="${a.command}">${a.label}</button>`).join('') +
        '</div>';
    }

    msgDiv.innerHTML = `<div class="message-bubble">${formatted}${actionsHtml}</div>`;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    msgDiv.querySelectorAll('.chat-action-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        chatInput.value = btn.getAttribute('data-cmd');
        sendChatMessage();
      });
    });
  }

  function appendTypingIndicator() {
    const id = 'typing_' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `<div class="message-bubble" style="color: #94a3b8; font-style: italic;">Assistant processing...</div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  // ---------------------------------------------------------
  // 7. Benchmark Analytics & Charts
  // ---------------------------------------------------------
  async function loadAnalytics() {
    if (!state.activeProject) return;
    try {
      const res = await fetch(`/api/analytics/${state.activeProject}`);
      const data = await res.json();
      const container = document.getElementById('analyticsTableContainer');

      if (!data.has_data) {
        if (container) container.innerHTML = '<p class="text-muted">No benchmark or finetuning results found yet. Click "Download All" to fetch remote outputs.</p>';
        return;
      }

      if (data.summary && data.summary.length > 0) {
        renderAnalyticsTable(data.summary);
        renderAnalyticsCharts(data.summary);
      } else if (container) {
        container.innerHTML = '<p class="text-muted">No summary.csv files found. Download remote results to populate.</p>';
      }

      renderFinetuneComparison(data.comparison);
      renderFinetuneStatusTable(data.finetune);
    } catch (err) {
      console.error(err);
    }
  }

  function renderFinetuneComparison(comparison) {
    const container = document.getElementById('finetuneComparisonContainer');
    if (!container) return;

    if (!comparison || comparison.length === 0) {
      container.innerHTML = '<p class="text-muted">Head-to-head comparison between pretrained paddleocr and fine-tuned paddleocr_ft will appear here once finetuning artifacts are collected.</p>';
      return;
    }

    let html = `
      <table class="custom-table">
        <thead>
          <tr>
            <th>Performance Metric</th>
            <th>Pretrained Baseline (PaddleOCR)</th>
            <th>Finetuned 10-Epochs (PaddleOCR-FT)</th>
            <th>Impact & Delta</th>
          </tr>
        </thead>
        <tbody>
    `;

    comparison.forEach(row => {
      let impactBadge = '';
      const diffStr = row.difference || 'N/A';
      if (diffStr.includes('[Improved]')) {
        impactBadge = `<span class="account-badge badge-active">${diffStr}</span>`;
      } else if (diffStr.includes('[Degraded]')) {
        impactBadge = `<span class="account-badge badge-error">${diffStr}</span>`;
      } else {
        impactBadge = `<code>${diffStr}</code>`;
      }

      html += `
        <tr>
          <td><strong>${row.metric || ''}</strong></td>
          <td><code>${row.pretrained || 'N/A'}</code></td>
          <td><code>${row.finetuned || 'N/A'}</code></td>
          <td>${impactBadge}</td>
        </tr>
      `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function renderFinetuneStatusTable(finetuneList) {
    const container = document.getElementById('finetuneStatusContainer');
    if (!container) return;

    if (!finetuneList || finetuneList.length === 0) {
      container.innerHTML = '<p class="text-muted">No finetuning execution status logs found. Run a finetune thread to train on DocLayNet.</p>';
      return;
    }

    let html = `
      <table class="custom-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Status</th>
            <th>Epochs</th>
            <th>Train Samples</th>
            <th>Val Samples</th>
            <th>Diagnostics / Details</th>
          </tr>
        </thead>
        <tbody>
    `;

    finetuneList.forEach(row => {
      let badgeClass = 'badge-secondary';
      if (row.status === 'trained' || row.status === 'ok') badgeClass = 'badge-complete';
      else if (row.status === 'failed' || row.status === 'error') badgeClass = 'badge-error';
      else if (row.status === 'prepared') badgeClass = 'badge-active';

      html += `
        <tr>
          <td><strong>${row.model || 'N/A'}</strong></td>
          <td><span class="account-badge ${badgeClass}">${row.status || 'N/A'}</span></td>
          <td>${row.epochs || 10}</td>
          <td>${row.train_samples || 0}</td>
          <td>${row.val_samples || 0}</td>
          <td style="font-size: 0.8rem; color: #94a3b8; max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${row.note || ''}">${row.note || 'N/A'}</td>
        </tr>
      `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function renderAnalyticsTable(summary) {
    const container = document.getElementById('analyticsTableContainer');
    let html = `
      <table class="custom-table">
        <thead>
          <tr>
            <th>Engine</th>
            <th>Status</th>
            <th>Success Pages</th>
            <th>CER (Char Error)</th>
            <th>WER (Word Error)</th>
            <th>Char F1</th>
            <th>Latency/Page (s)</th>
            <th>Chars / Sec</th>
          </tr>
        </thead>
        <tbody>
    `;

    summary.forEach(row => {
      html += `
        <tr>
          <td><strong>${row.engine || 'N/A'}</strong></td>
          <td><span class="account-badge ${row.status === 'ok' ? 'badge-complete' : 'badge-error'}">${row.status || 'N/A'}</span></td>
          <td>${row.success_pages || 0} / 20</td>
          <td><code>${(row.cer !== null && row.cer !== undefined) ? Number(row.cer).toFixed(4) : 'NaN'}</code></td>
          <td><code>${(row.wer !== null && row.wer !== undefined) ? Number(row.wer).toFixed(4) : 'NaN'}</code></td>
          <td><code>${(row.char_f1 !== null && row.char_f1 !== undefined) ? Number(row.char_f1).toFixed(4) : 'NaN'}</code></td>
          <td>${(row.latency_s !== null && row.latency_s !== undefined) ? Number(row.latency_s).toFixed(2) + 's' : 'NaN'}</td>
          <td><strong>${(row.chars_per_second !== null && row.chars_per_second !== undefined) ? Number(row.chars_per_second).toFixed(1) : 'NaN'}</strong></td>
        </tr>
      `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function renderAnalyticsCharts(summary) {
    const validRows = summary.filter(r => r.cer !== null && !isNaN(r.cer));
    const labels = validRows.map(r => r.engine);
    const cerData = validRows.map(r => r.cer);
    const werData = validRows.map(r => r.wer);
    const speedData = validRows.map(r => r.chars_per_second || 0);

    // Chart 1: Errors (CER & WER)
    const ctxErr = document.getElementById('chartErrors').getContext('2d');
    if (state.chartErrorsInstance) state.chartErrorsInstance.destroy();
    state.chartErrorsInstance = new Chart(ctxErr, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'CER (Character)', data: cerData, backgroundColor: 'rgba(56, 189, 248, 0.8)' },
          { label: 'WER (Word)', data: werData, backgroundColor: 'rgba(129, 140, 248, 0.8)' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#f8fafc' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // Chart 2: Speed (Chars / Second)
    const ctxSpeed = document.getElementById('chartSpeed').getContext('2d');
    if (state.chartSpeedInstance) state.chartSpeedInstance.destroy();
    state.chartSpeedInstance = new Chart(ctxSpeed, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Characters / Second', data: speedData, backgroundColor: 'rgba(52, 211, 153, 0.8)' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#f8fafc' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }

  // ---------------------------------------------------------
  // 8. Live Logs & Debugging
  // ---------------------------------------------------------
  async function loadLogs() {
    if (!state.activeProject) return;
    const tailLines = logTailLines.value || 100;
    try {
      const res = await fetch(`/api/logs/${state.activeProject}?file_path=${encodeURIComponent(state.selectedLogFile)}&max_lines=${tailLines}`);
      const data = await res.json();

      logFileSelect.innerHTML = '';
      if (!data.files || data.files.length === 0) {
        logFileSelect.innerHTML = '<option value="">No log files available</option>';
        logOutputContainer.textContent = 'No log files downloaded yet.';
        return;
      }

      data.files.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.path;
        opt.textContent = `${f.job} / ${f.name}`;
        if (f.path === data.selected_file) opt.selected = true;
        logFileSelect.appendChild(opt);
      });

      state.selectedLogFile = data.selected_file;
      currentLogName.textContent = data.selected_file.split(/[\\/]/).pop();
      renderLogContent(data.content);
    } catch (err) {
      console.error(err);
    }
  }

  function renderLogContent(rawText) {
    if (!rawText) {
      logOutputContainer.textContent = 'Empty log file.';
      return;
    }
    const filter = logSearchInput.value.toLowerCase().trim();
    if (!filter) {
      logOutputContainer.textContent = rawText;
      return;
    }
    const lines = rawText.split('\n');
    const matched = lines.filter(l => l.toLowerCase().includes(filter));
    logOutputContainer.textContent = matched.length > 0 ? matched.join('\n') : `No lines matched "${filter}".`;
  }

  logFileSelect.addEventListener('change', () => {
    state.selectedLogFile = logFileSelect.value;
    loadLogs();
  });
  logTailLines.addEventListener('change', () => loadLogs());
  btnRefreshLog.addEventListener('click', () => loadLogs());
  logSearchInput.addEventListener('input', () => {
    const raw = logOutputContainer.getAttribute('data-raw') || logOutputContainer.textContent;
    logOutputContainer.setAttribute('data-raw', raw);
    renderLogContent(raw);
  });

  // ---------------------------------------------------------
  // 9. Accounts & Git Management
  // ---------------------------------------------------------
  formAddAccount.addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = document.getElementById('newAccId').value.trim();
    const username = document.getElementById('newAccUser').value.trim();
    if (!id || !username) return;

    try {
      const res = await fetch('/api/accounts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, username }),
      });
      const data = await res.json();
      if (data.success) {
        showToast(data.message, 'success');
        formAddAccount.reset();
        loadAccounts();
      } else {
        showToast(data.detail || 'Failed to add account.', 'error');
      }
    } catch (err) {
      showToast('Connection error: ' + err.message, 'error');
    }
  });

  btnUploadKaggleJson.addEventListener('click', async () => {
    const accId = tokenAccountSelect.value;
    const file = fileKaggleJson.files[0];
    if (!file) {
      showToast('Please select a kaggle.json file first.', 'error');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`/api/accounts/${accId}/upload_kaggle_json`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        showToast('kaggle.json uploaded successfully.', 'success');
        loadAccounts();
      }
    } catch (err) {
      showToast('Upload error: ' + err.message, 'error');
    }
  });

  btnSaveApiKey.addEventListener('click', async () => {
    const accId = tokenAccountSelect.value;
    const token = inputApiKey.value.trim();
    if (!token) return;

    try {
      const res = await fetch(`/api/accounts/${accId}/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      const data = await res.json();
      if (data.success) {
        showToast('API Token saved.', 'success');
        inputApiKey.value = '';
        loadAccounts();
      }
    } catch (err) {
      showToast('Save token error: ' + err.message, 'error');
    }
  });

  btnValidateToken.addEventListener('click', async () => {
    const accId = tokenAccountSelect.value;
    tokenValidationResult.innerHTML = '<span class="text-cyan">Validating token...</span>';
    try {
      const res = await fetch(`/api/accounts/${accId}/validate`, { method: 'POST' });
      const data = await res.json();
      if (data.ok) {
        tokenValidationResult.innerHTML = `<span style="color: #34d399;">Token valid: <b>${data.token_username}</b> (${data.key_len} chars)</span>`;
      } else {
        tokenValidationResult.innerHTML = `<span style="color: #f87171;">Token invalid: ${data.message}</span>`;
      }
    } catch (err) {
      tokenValidationResult.innerHTML = `<span style="color: #f87171;">Error: ${err.message}</span>`;
    }
  });

  btnGitStatus.addEventListener('click', async () => {
    gitStatusOutput.textContent = 'Checking git status...';
    try {
      const res = await fetch('/api/git/status');
      const data = await res.json();
      gitStatusOutput.textContent = `Branch: ${data.branch}\n\n${data.status || 'Clean working tree.'}`;
    } catch (err) {
      gitStatusOutput.textContent = 'Error: ' + err.message;
    }
  });

  btnGitPush.addEventListener('click', async () => {
    showToast('Pushing git commits to remote repository...', 'info');
    try {
      const res = await fetch('/api/git/push', { method: 'POST' });
      const data = await res.json();
      gitStatusOutput.textContent = data.output;
      showToast('Git push completed.', 'success');
    } catch (err) {
      showToast('Git push error: ' + err.message, 'error');
    }
  });

  // ---------------------------------------------------------
  // 10. Toast Notification System
  // ---------------------------------------------------------
  function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function setupEventListeners() {
    projectSelector.addEventListener('change', () => {
      state.activeProject = projectSelector.value;
      loadJobsStatus();
      loadAnalytics();
      loadLogs();
    });
  }

  // Initialize
  init();
});
