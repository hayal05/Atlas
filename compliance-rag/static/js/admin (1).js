let token = sessionStorage.getItem('adminToken') || '';

const gate = document.getElementById('gate');
const main = document.getElementById('main');
const gateStatusMsg = document.getElementById('gate-status-msg');
const mainStatusMsg = document.getElementById('main-status-msg');
const tokenInput = document.getElementById('token');

function setStatus(msg, isError) {
  const target = (main.style.display === 'block') ? mainStatusMsg : gateStatusMsg;
  [gateStatusMsg, mainStatusMsg].forEach(el => { el.textContent = ''; el.className = 'status-msg'; });
  target.textContent = msg || '';
  target.className = 'status-msg' + (msg ? (isError ? ' error' : ' ok') : '');
}

async function authedFetch(path, opts = {}) {
  opts.headers = Object.assign({}, opts.headers, { 'X-Admin-Token': token });
  const res = await fetch(path, opts);
  if (res.status === 401 || res.status === 503) {
    sessionStorage.removeItem('adminToken');
    token = '';
    showGate(res.status === 503
      ? 'Admin access is not configured on this deployment (set ADMIN_TOKEN).'
      : 'Invalid token.');
    throw new Error('unauthorized');
  }
  return res;
}

function showGate(msg) {
  gate.style.display = 'block';
  main.style.display = 'none';
  if (msg) setStatus(msg, true);
}

function showMain() {
  gate.style.display = 'none';
  main.style.display = 'block';
  loadDocuments();
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function loadDocuments() {
  try {
    const res = await authedFetch('/api/admin/documents');
    const data = await res.json();
    document.getElementById('chunkCount').textContent = data.indexed_chunks + ' chunks indexed';
    const rows = document.getElementById('docRows');
    if (!data.documents.length) {
      rows.innerHTML = '<tr class="empty-row"><td colspan="3">No documents yet — upload one above.</td></tr>';
      return;
    }
    rows.innerHTML = data.documents.map(d => `
      <tr>
        <td>${escapeHtml(d.filename)}</td>
        <td class="size">${fmtSize(d.size_bytes)}</td>
        <td><button class="del" data-name="${escapeHtml(d.filename)}">Remove</button></td>
      </tr>
    `).join('');
    rows.querySelectorAll('.del').forEach(btn => {
      btn.addEventListener('click', () => removeDoc(btn.dataset.name));
    });
  } catch (e) { /* handled in authedFetch */ }
}

async function removeDoc(filename) {
  if (!confirm(`Remove "${filename}" and reindex?`)) return;
  try {
    await authedFetch('/api/admin/documents/' + encodeURIComponent(filename), { method: 'DELETE' });
    loadDocuments();
  } catch (e) {}
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  setStatus(`Uploading ${file.name}…`, false);
  try {
    const res = await authedFetch('/api/admin/documents', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setStatus(err.detail || 'Upload failed.', true);
      return;
    }
    setStatus(`Uploaded ${file.name}.`, false);
    loadDocuments();
  } catch (e) {
    if (e.message !== 'unauthorized') {
      setStatus(`Upload failed: ${e.message || 'network error, please retry.'}`, true);
    }
  }
}

function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

document.getElementById('unlock').addEventListener('click', () => {
  const t = tokenInput.value.trim();
  if (!t) return;
  token = t;
  sessionStorage.setItem('adminToken', t);
  setStatus('', false);
  showMain();
});
tokenInput.addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('unlock').click(); });

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); fileInput.value = ''; });
['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

if (token) showMain(); else showGate();
