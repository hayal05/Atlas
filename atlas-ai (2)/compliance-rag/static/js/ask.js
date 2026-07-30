const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let first = true;

function addMessage(role, html, grounded) {
  if (first) { log.innerHTML = ''; first = false; }
  const div = document.createElement('div');
  div.className = 'msg ' + role + (grounded === false ? ' ungrounded' : '');
  div.innerHTML = `<div class="role">${role === 'user' ? 'You' : 'Assistant'}</div>
                    <div class="bubble">${html}</div>`;
  log.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage('user', escapeHtml(question));
  input.value = '';
  sendBtn.disabled = true;

  const thinking = addMessage('assistant', '<p><em>Checking procedure documents…</em></p>');

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();

    let html = '<p>' + escapeHtml(data.answer).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>') + '</p>';
    if (data.citations && data.citations.length) {
      html += '<div class="citations">' + data.citations.map(c =>
        `<span class="citation"><b>${escapeHtml(c.source)}</b> · ${escapeHtml(c.heading)} (${c.relevance})</span>`
      ).join('') + '</div>';
    }
    thinking.querySelector('.bubble').innerHTML = html;
    thinking.className = 'msg assistant' + (data.grounded ? '' : ' ungrounded');
  } catch (err) {
    thinking.querySelector('.bubble').innerHTML = '<p>Something went wrong reaching the assistant. Please try again.</p>';
  } finally {
    sendBtn.disabled = false;
  }
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
