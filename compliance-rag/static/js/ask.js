const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const hamburger = document.getElementById('hamburger');
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('overlay');
const historyList = document.getElementById('historyList');
const newChatBtn = document.getElementById('newChatBtn');

// Chat history is kept client-side (this browser only) in localStorage --
// there's no login system on this app, so there's no server-side place to
// attribute conversations to a person. Nothing question/answer-related is
// sent anywhere new; this just persists what's already on screen so it
// survives a reload and lets you switch between past conversations.
const STORAGE_KEY = 'atlas_chats_v1';
const ACTIVE_KEY = 'atlas_active_chat_v1';
const MOBILE_BREAKPOINT = 860;

function uid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'c' + Date.now() + Math.random().toString(16).slice(2);
}

function loadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveChats() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch {
    // Storage full or unavailable (e.g. private browsing) -- fail silently,
    // the current session still works, it just won't persist.
  }
}

let chats = loadChats();
let activeId = localStorage.getItem(ACTIVE_KEY);

function getActiveChat() {
  return chats.find((c) => c.id === activeId);
}

function setActive(id) {
  activeId = id;
  localStorage.setItem(ACTIVE_KEY, id);
}

function createChat() {
  const chat = { id: uid(), title: 'New chat', messages: [], updatedAt: Date.now() };
  chats.unshift(chat);
  saveChats();
  setActive(chat.id);
  return chat.id;
}

if (!activeId || !getActiveChat()) {
  setActive(chats.length ? chats[0].id : createChat());
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const MODE_BADGE = {
  grounded: '',
  grounded_partial: '<div class="badge badge-partial">Partial match in policy documents</div>',
  general: '<div class="badge badge-general">General knowledge · not from your policy documents</div>',
  unavailable: '<div class="badge badge-unavailable">No matching policy found</div>',
};

// Renders the answer body, including the References footer the backend
// appends to grounded/partial answers (plain "---" divider + "**bold**").
function renderAnswer(text) {
  let html = escapeHtml(text);
  html = html.replace(/^---$/gm, '<hr class="ref-divider">');
  html = html.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  html = html.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
  return '<p>' + html + '</p>';
}

function renderMessageDom(role, html, mode) {
  const div = document.createElement('div');
  div.className = 'msg ' + role + (mode && mode !== 'grounded' ? ' ' + mode : '');
  div.innerHTML = `<div class="role">${role === 'user' ? 'You' : 'Assistant'}</div>
                    <div class="bubble">${html}</div>`;
  log.appendChild(div);
  return div;
}

function renderChatIntoLog(chat) {
  log.innerHTML = '';
  if (!chat || !chat.messages.length) {
    log.innerHTML = '<div class="empty">Try: "What receipts do I need for a $200 client dinner?"</div>';
    return;
  }
  chat.messages.forEach((m) => renderMessageDom(m.role, m.html, m.mode));
  window.scrollTo(0, document.body.scrollHeight);
}

function chatTitleFrom(html) {
  const text = html.replace(/<[^>]+>/g, '').trim();
  return text.length > 42 ? text.slice(0, 42).trim() + '…' : (text || 'New chat');
}

function persistMessage(role, html, mode) {
  const chat = getActiveChat();
  if (!chat) return;
  chat.messages.push({ role, html, mode });
  if (chat.title === 'New chat' && role === 'user') {
    chat.title = chatTitleFrom(html);
  }
  chat.updatedAt = Date.now();
  chats.sort((a, b) => b.updatedAt - a.updatedAt);
  saveChats();
  renderHistoryList();
}

function renderHistoryList() {
  historyList.innerHTML = '';
  if (!chats.length) {
    historyList.innerHTML = '<div class="history-empty">No conversations yet.</div>';
    return;
  }
  chats.forEach((chat) => {
    const item = document.createElement('div');
    item.className = 'history-item' + (chat.id === activeId ? ' active' : '');

    const titleBtn = document.createElement('button');
    titleBtn.className = 'history-title';
    titleBtn.type = 'button';
    titleBtn.textContent = chat.title || 'New chat';
    titleBtn.addEventListener('click', () => {
      setActive(chat.id);
      renderChatIntoLog(chat);
      renderHistoryList();
      closeSidebarOnMobile();
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'history-delete';
    delBtn.type = 'button';
    delBtn.title = 'Delete conversation';
    delBtn.setAttribute('aria-label', 'Delete conversation');
    delBtn.textContent = '×';
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!confirm('Delete this conversation? This can\'t be undone.')) return;
      deleteChat(chat.id);
    });

    item.appendChild(titleBtn);
    item.appendChild(delBtn);
    historyList.appendChild(item);
  });
}

function deleteChat(id) {
  chats = chats.filter((c) => c.id !== id);
  saveChats();
  if (activeId === id) {
    const nextId = chats.length ? chats[0].id : createChat();
    setActive(nextId);
    renderChatIntoLog(getActiveChat());
  }
  renderHistoryList();
}

function openSidebar() {
  sidebar.classList.add('open');
  overlay.classList.add('show');
  hamburger.setAttribute('aria-expanded', 'true');
}

function closeSidebar() {
  sidebar.classList.remove('open');
  overlay.classList.remove('show');
  hamburger.setAttribute('aria-expanded', 'false');
}

function closeSidebarOnMobile() {
  if (window.innerWidth <= MOBILE_BREAKPOINT) closeSidebar();
}

hamburger.addEventListener('click', () => {
  sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
});
overlay.addEventListener('click', closeSidebar);

newChatBtn.addEventListener('click', () => {
  setActive(createChat());
  renderChatIntoLog(getActiveChat());
  renderHistoryList();
  closeSidebarOnMobile();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  const userHtml = escapeHtml(question);
  renderMessageDom('user', userHtml);
  persistMessage('user', userHtml);
  input.value = '';
  sendBtn.disabled = true;

  const thinking = renderMessageDom('assistant', '<div class="thinking"><img src="/static/img/logo.png" class="thinking-logo" alt="" /><span>Checking procedure documents…</span></div>');
  window.scrollTo(0, document.body.scrollHeight);

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    const mode = data.mode || (data.grounded ? 'grounded' : 'unavailable');
    let html = (MODE_BADGE[mode] || '') + renderAnswer(data.answer);
    if (data.citations && data.citations.length) {
      html += '<div class="citations">' + data.citations.map(c => {
        const loc = (c.page ? `p.${c.page} · ` : '') + `#${c.chunk_number}`;
        return `<span class="citation"><b>${escapeHtml(c.source)}</b> · ${escapeHtml(c.heading)} · ${loc} (${c.relevance})</span>`;
      }).join('') + '</div>';
    }
    thinking.querySelector('.bubble').innerHTML = html;
    thinking.className = 'msg assistant' + (mode !== 'grounded' ? ' ' + mode : '');
    persistMessage('assistant', html, mode);
  } catch (err) {
    const errHtml = '<p>Something went wrong reaching the assistant. Please try again.</p>';
    thinking.querySelector('.bubble').innerHTML = errHtml;
    persistMessage('assistant', errHtml, 'unavailable');
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

// Initial paint
renderChatIntoLog(getActiveChat());
renderHistoryList();
