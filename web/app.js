/* ============================================================
   前端逻辑：三栏界面 + SSE 流式渲染

   没有用任何框架。原因是这个界面只有三块内容、四个交互动作，
   引入构建工具和框架反而要花时间去解释「为什么要它」——
   而这个项目要拿去答辩，每个文件都得能讲清楚。

   交互主线：
     填目标 → 建会话 → 拉 SSE（讲解逐字冒出来）→ 看到题目
     → 提交答案 → 看到批改 → 路径和记忆面板跟着变
   ============================================================ */

const el = {
  goalForm: document.getElementById('goal-form'),
  goalInput: document.getElementById('goal-input'),
  userInput: document.getElementById('user-input'),
  startBtn: document.getElementById('start-btn'),
  statusChip: document.getElementById('status-chip'),
  statusText: document.getElementById('status-text'),
  spinner: document.getElementById('spinner'),
  alert: document.getElementById('alert'),
  alertText: document.getElementById('alert-text'),
  alertClose: document.getElementById('alert-close'),
  chat: document.getElementById('chat'),
  answerForm: document.getElementById('answer-form'),
  answerInput: document.getElementById('answer-input'),
  answerBtn: document.getElementById('answer-btn'),
  questionHint: document.getElementById('question-hint'),
  path: document.getElementById('path'),
  pathProgress: document.getElementById('path-progress'),
  profile: document.getElementById('profile'),
  events: document.getElementById('events'),
  facts: document.getElementById('facts'),
  memoryStats: document.getElementById('memory-stats'),
  eventCount: document.getElementById('event-count'),
  factCount: document.getElementById('fact-count'),
};

const STATUS_LABEL = { focus: '重点攻', practice: '多练习', review: '快速过' };

let sessionId = null;
let busy = false;
let currentTopic = '';
let currentQuestionText = '';
let lastExplanation = '';
let streamNode = null; /* 正在逐字渲染的那个气泡 */
let eventSource = null;

/* ------------------------------------------------------------
   小工具
   ------------------------------------------------------------ */

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** 调用后端。出错时把后端给的中文说明抛出来，而不是只报一个状态码。 */
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data = {};
  try {
    data = await response.json();
  } catch {
    /* 后端没返回 JSON（比如 500 的默认页），下面统一按状态码处理 */
  }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return data;
}

function setStatus(text, spinning = false) {
  el.statusChip.hidden = false;
  el.statusText.textContent = text;
  el.spinner.hidden = !spinning;
}

function showError(message) {
  el.alertText.textContent = message;
  el.alert.hidden = false;
}

function hideError() {
  el.alert.hidden = true;
}

/** 忙碌时锁住输入，避免连点产生两次请求。 */
function setBusy(value) {
  busy = value;
  el.startBtn.disabled = value;
  el.goalInput.disabled = value;
  el.userInput.disabled = value;
  const canAnswer = !value && Boolean(sessionId) && Boolean(currentQuestionText);
  el.answerInput.disabled = !canAnswer;
  el.answerBtn.disabled = !canAnswer;
}

function scrollChatToBottom() {
  el.chat.scrollTop = el.chat.scrollHeight;
}

/* ------------------------------------------------------------
   对话气泡
   ------------------------------------------------------------ */

function addBubble(kind, role, text) {
  const node = document.createElement('div');
  node.className = `bubble ${kind}`;
  if (role) {
    const head = document.createElement('span');
    head.className = 'bubble-role';
    head.textContent = role;
    node.appendChild(head);
  }
  const body = document.createElement('span');
  body.className = 'bubble-body';
  body.textContent = text;
  node.appendChild(body);
  el.chat.appendChild(node);
  scrollChatToBottom();
  return node;
}

function addSystemNote(text) {
  return addBubble('system', '', text);
}

/**
 * 渲染带代码块的文字，并在末尾挂一个闪烁光标。
 *
 * 导师的讲解里常有 ``` 包起来的示例代码，直接当纯文本插入会丢格式；
 * 但也不能把整段按 HTML 插进去——那是模型生成的文字，不该被当成标记解析。
 * 折中做法：只有自己包出来的 <pre> 是标签，其余内容一律转义。
 */
function renderRich(node, text, withCursor) {
  const parts = String(text).split('```');
  let html = '';
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const body = part.replace(/^[a-zA-Z+#]*\n/, '');
      html += `<pre class="code">${escapeHtml(body.replace(/\n$/, ''))}</pre>`;
    } else {
      html += escapeHtml(part);
    }
  });
  node.innerHTML = html + (withCursor ? '<span class="cursor"></span>' : '');
}

/* ------------------------------------------------------------
   开课
   ------------------------------------------------------------ */

async function startLearning(event) {
  event.preventDefault();
  if (busy) return;

  const goal = el.goalInput.value.trim();
  if (!goal) {
    showError('请先填写学习目标');
    return;
  }

  hideError();
  el.chat.innerHTML = '';
  el.path.innerHTML = '';
  el.profile.textContent = '尚未建立';
  el.events.innerHTML = '';
  el.facts.innerHTML = '';
  el.memoryStats.textContent = '—';
  currentTopic = '';
  currentQuestionText = '';
  lastExplanation = '';
  streamNode = null;

  setBusy(true);
  setStatus('正在创建会话……', true);

  try {
    const info = await api('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, user_id: el.userInput.value.trim() || 'demo-user' }),
    });
    sessionId = info.session_id;
    addSystemNote(`会话 ${sessionId} 已创建 · 目标「${goal}」`);
    setStatus('正在拆解学习目标……', true);
    openStartStream();
  } catch (error) {
    setBusy(false);
    setStatus('出错了', false);
    showError(`创建会话失败：${error.message}`);
  }
}

function openStartStream() {
  if (eventSource) eventSource.close();

  eventSource = new EventSource(`/api/sessions/${sessionId}/start/stream`);

  eventSource.addEventListener('status', (event) => {
    const data = JSON.parse(event.data);
    addSystemNote(data.message);
  });

  eventSource.addEventListener('token', (event) => {
    const data = JSON.parse(event.data);
    if (!streamNode) {
      streamNode = addBubble('tutor', '导师讲解', '');
    }
    streamNode.dataset.raw = (streamNode.dataset.raw || '') + data.text;
    renderRich(streamNode.querySelector('.bubble-body'), streamNode.dataset.raw, true);
    scrollChatToBottom();
  });

  eventSource.addEventListener('state', (event) => {
    const data = JSON.parse(event.data);
    if (streamNode) {
      renderRich(streamNode.querySelector('.bubble-body'), streamNode.dataset.raw, false);
      lastExplanation = streamNode.dataset.raw;
      streamNode = null;
    }
    applyState(data);
    setBusy(false);
    setStatus(data.finished ? '已完成' : '等待作答', false);
    /* 开课就已经写了不少记忆（讲解、布置题目都算事件），
       这里不刷新的话右栏要等到第一次作答才亮起来 */
    refreshSidePanels();
  });

  eventSource.addEventListener('error', (event) => {
    /* 后端的 error 事件带说明文字；连接自身的错误则没有 data */
    if (event.data) {
      const data = JSON.parse(event.data);
      showError(`开课失败：${data.message}`);
      setStatus('出错了', false);
      setBusy(false);
    }
    closeStream();
  });

  eventSource.addEventListener('done', () => closeStream());
}

function closeStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

/* ------------------------------------------------------------
   作答
   ------------------------------------------------------------ */

async function submitAnswer(event) {
  event.preventDefault();
  const answer = el.answerInput.value.trim();
  if (!answer || !sessionId || busy) return;

  hideError();
  setBusy(true);
  addBubble('student', '我的作答', answer);
  el.answerInput.value = '';
  el.questionHint.textContent = '';
  setStatus('正在批改……', true);

  const topicBefore = currentTopic;

  try {
    const data = await api(`/api/sessions/${sessionId}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer }),
    });

    if (data.grade) {
      addBubble(
        data.grade.correct ? 'grade-right' : 'grade-wrong',
        data.grade.correct ? '判对了' : '判错了',
        data.grade.feedback,
      );
    }

    if (data.explanation && data.explanation !== lastExplanation) {
      lastExplanation = data.explanation;
      addBubble('tutor', '导师换个讲法', data.explanation);
    }
    if (data.topic && topicBefore && data.topic !== topicBefore) {
      addSystemNote(`换知识点：《${topicBefore}》→《${data.topic}》`);
    }

    applyState(data);

    if (data.finished) {
      addSystemNote('这条学习路径走完了');
      setStatus('已完成', false);
    } else {
      setStatus('等待作答', false);
    }
  } catch (error) {
    showError(`提交失败：${error.message}`);
    setStatus('等待作答', false);
  } finally {
    /* 先松开锁，再按最新状态决定输入框是否可用 */
    busy = false;
    setBusy(false);
    refreshSidePanels();
  }
}

/* ------------------------------------------------------------
   渲染
   ------------------------------------------------------------ */

function applyState(data) {
  currentTopic = data.topic || '';
  renderPath(data);
  renderQuestion(data);
  setBusy(false);
}

function renderQuestion(data) {
  const question = data.question;
  if (!question) {
    currentQuestionText = '';
    if (data.finished) el.questionHint.textContent = '学习已完成';
    return;
  }

  /* 题目内容没变就不重复刷屏（比如重讲之后还可能是同一题） */
  const text = question.question;
  if (text === currentQuestionText) return;
  currentQuestionText = text;

  addBubble('question', `第 ${question.index + 1} / ${question.total} 题`, text);
  el.questionHint.textContent = question.hint ? `提示：${question.hint}` : '';
}

function renderPath(data) {
  const steps = (data.path && data.path.steps) || [];
  if (!steps.length) {
    el.path.innerHTML = '<p class="empty-note">还没有知识点，先填个目标开始学习</p>';
    el.pathProgress.textContent = '未开始';
    return;
  }

  const progress = data.progress || {};
  el.pathProgress.textContent = `${progress.done || 0}/${progress.total || 0} 已掌握`;

  el.path.innerHTML = '';
  steps.forEach((step) => {
    const ratio = Math.max(0, Math.min(1, step.mastery || 0));
    const level = ratio >= 0.7 ? 'high' : ratio >= 0.4 ? 'mid' : 'low';
    const isCurrent = step.topic === data.topic;

    const card = document.createElement('div');
    card.className = 'topic-card'
      + (isCurrent ? ' current' : '')
      + (ratio >= 0.7 ? ' mastered' : '')
      + (step.depends_on && step.depends_on.length ? ' has-deps' : '');

    card.innerHTML = `
      <div class="topic-head">
        <span class="topic-name">${escapeHtml(step.topic)}</span>
        <span class="topic-index">${step.order}</span>
      </div>
      <div class="mastery-bar"><div class="mastery-fill ${level}" style="width:${ratio * 100}%"></div></div>
      <div class="topic-meta">
        <span class="tag ${step.status}">${STATUS_LABEL[step.status] || step.status}</span>
        <span>掌握度 ${ratio.toFixed(2)}${isCurrent ? ' · 正在学' : ''}</span>
      </div>
      ${step.depends_on && step.depends_on.length
        ? `<div class="topic-deps">前置：${escapeHtml(step.depends_on.join('、'))}</div>`
        : ''}
    `;
    el.path.appendChild(card);
  });
}

async function refreshSidePanels() {
  if (!sessionId) return;
  try {
    const [profile, memory] = await Promise.all([
      api(`/api/sessions/${sessionId}/profile`),
      api(`/api/sessions/${sessionId}/memory`),
    ]);
    el.profile.textContent = profile.snapshot || '尚未建立';
    renderMemory(memory);
  } catch {
    /* 侧栏是辅助信息，拉不到就不打断主流程 */
  }
}

function renderMemory(memory) {
  const stats = memory.stats || {};
  el.memoryStats.textContent = `对话 ${stats.messages} · 事件 ${stats.events} · 事实 ${stats.facts}`;

  el.eventCount.textContent = `(${memory.events.length})`;
  el.events.innerHTML = memory.events.length
    ? memory.events
        .slice(0, 12)
        .map((item) => `
          <li class="${item.importance >= 0.7 ? 'important' : ''}">
            <span class="event-type">${escapeHtml(item.event_type)}</span>
            ${escapeHtml(item.content)}
          </li>`)
        .join('')
    : '<li>还没有事件</li>';

  el.factCount.textContent = `(${memory.facts.length})`;
  el.facts.innerHTML = memory.facts.length
    ? memory.facts
        .slice(0, 12)
        .map((item) => `
          <li>
            <span class="event-type">${escapeHtml(item.category)}</span>
            ${escapeHtml(item.key)} = ${escapeHtml(item.value)}
            <span style="color:var(--text-faint)">（${item.confidence.toFixed(1)}）</span>
          </li>`)
        .join('')
    : '<li>还没有沉淀出事实</li>';
}

/* ------------------------------------------------------------
   事件绑定
   ------------------------------------------------------------ */

el.goalForm.addEventListener('submit', startLearning);
el.answerForm.addEventListener('submit', submitAnswer);
el.alertClose.addEventListener('click', hideError);

/* Ctrl/Cmd + Enter 快捷提交作答 */
el.answerInput.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    el.answerForm.requestSubmit();
  }
});
