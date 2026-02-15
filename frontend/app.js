// ========================================
// PaperMind - 前端逻辑
// ========================================

// API配置
// const API_BASE_URL = '/api';
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
    chat: `${API_BASE_URL}/chat`,
    upload: `${API_BASE_URL}/upload-pdf`,
    modes: `${API_BASE_URL}/modes`,
    pdf: `${API_BASE_URL}/pdf`,
    reset: `${API_BASE_URL}/reset-agent`,
    getProcessMessages: `${API_BASE_URL}/get-process-messages`,
    clearProcessMessages: `${API_BASE_URL}/clear-process-messages`
};

// 应用状态
const appState = {
    messages: [
        { role: 'assistant', content: '您好，我是您的智能文献助手，可以帮助您检索、分析和解答学术文献相关问题。' }
    ],
    currentMode: 'existing',
    currentPdfPath: null,
    uploadedPdfPath: null,
    threadId: generateSessionId(),
    isProcessing: false,
    processPollingInterval: null
};

// DOM元素引用
const elements = {
    // 表单和输入
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    clearBtn: document.getElementById('clear-btn'),
    
    // 聊天区域
    chatContent: document.getElementById('chat-content'),
    chatContainer: document.getElementById('chat-container'),
    chatStats: document.getElementById('chat-stats'),
    
    // 处理过程
    processContent: document.getElementById('process-container'),
    processContainer: document.getElementById('process-container'),
    clearProcessBtn: document.getElementById('clear-process-btn'),
    loadingAnimation: document.getElementById('loading-animation'),
    
    // 模式选择
    modeSelect: document.getElementById('mode-select'),
    modeOptions: document.querySelectorAll('.mode-option'),
    
    // 上传区域
    uploadPanel: document.getElementById('upload-panel'),
    pdfUpload: document.getElementById('pdf-upload'),
    uploadDropzone: document.getElementById('upload-dropzone'),
    uploadStatus: document.getElementById('upload-status'),
    
    // PDF预览
    pdfContainer: document.getElementById('pdf-container'),
    pdfInfo: document.getElementById('pdf-info'),
    
    // API-KEY设置
    apiKeyInput: document.getElementById('api-key-input'),
    apiKeySubmit: document.getElementById('api-key-submit'),
    apiKeyStatus: document.getElementById('api-key-status'),
    clearApiKeyStatusBtn: document.getElementById('clear-api-key-status-btn'),
    
    // 系统状态
    statusIndicator: document.querySelector('.status-indicator'),
    statusText: document.querySelector('.status-text'),
    
    // 公告栏
    announcementContainer: document.getElementById('announcement-container'),
    announcementModal: document.getElementById('announcement-modal'),
    announcementModalOverlay: document.getElementById('announcement-modal-overlay'),
    announcementModalClose: document.getElementById('announcement-modal-close'),
    announcementModalTitle: document.getElementById('announcement-modal-title'),
    announcementModalContent: document.getElementById('announcement-modal-content')
};

// ========================================
// 初始化应用
// ========================================
function initApp() {
    console.log('PaperMind 正在初始化...');
    setupEventListeners();
    loadDefaultPdf();
    updateChatStats();
    updateProcessMessages();
    checkBackendConnection();
    
    console.log('PaperMind 初始化完成');
}

// ========================================
// 事件监听器设置
// ========================================
function setupEventListeners() {
    // 聊天表单提交
    elements.chatForm.addEventListener('submit', handleChatSubmit);
    
    // 输入框键盘事件（Enter发送，Ctrl+Enter换行）
    elements.chatInput.addEventListener('keydown', handleInputKeydown);
    
    // 清空按钮
    elements.clearBtn.addEventListener('click', clearChat);
    
    // 模式选择 - 下拉框
    elements.modeSelect.addEventListener('change', handleModeChange);
    
    // 模式选择 - 单选按钮（保留用于兼容性）
    elements.modeOptions.forEach(option => {
        option.addEventListener('click', handleModeChange);
    });
    
    // PDF上传 - 点击上传区域
    elements.uploadDropzone.addEventListener('click', () => {
        elements.pdfUpload.click();
    });
    
    // PDF上传 - 文件选择
    elements.pdfUpload.addEventListener('change', handlePdfUpload);
    
    // PDF上传 - 拖拽支持
    elements.uploadDropzone.addEventListener('dragover', handleDragOver);
    elements.uploadDropzone.addEventListener('dragleave', handleDragLeave);
    elements.uploadDropzone.addEventListener('drop', handleDrop);
    
    // 清空过程消息按钮
    elements.clearProcessBtn.addEventListener('click', clearProcessMessages);
    
    // API-KEY设置
    elements.apiKeySubmit.addEventListener('click', handleApiKeySubmit);
    elements.apiKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleApiKeySubmit();
        }
    });
    
    // 清除API-KEY状态按钮
    if (elements.clearApiKeyStatusBtn) {
        elements.clearApiKeyStatusBtn.addEventListener('click', clearApiKeyStatus);
    }
    
    // 公告栏事件监听
    setupAnnouncementEventListeners();
}

// ========================================
// 处理输入框键盘事件
// ========================================
function handleInputKeydown(e) {
    // Ctrl+Enter: 换行
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        const start = elements.chatInput.selectionStart;
        const end = elements.chatInput.selectionEnd;
        const value = elements.chatInput.value;
        elements.chatInput.value = value.substring(0, start) + '\n' + value.substring(end);
        elements.chatInput.selectionStart = elements.chatInput.selectionEnd = start + 1;
        return;
    }
    
    // Enter: 发送消息（但不包含Ctrl+Enter的情况）
    if (e.key === 'Enter' && !e.ctrlKey) {
        e.preventDefault();
        handleChatSubmit(e);
    }
}

// ========================================
// 处理聊天提交
// ========================================
async function handleChatSubmit(e) {
    e.preventDefault();
    
    const message = elements.chatInput.value.trim();
    if (!message || appState.isProcessing) return;
    
    // 清空输入框
    elements.chatInput.value = '';
    
    // 检查模式特定的前置条件
    if (appState.currentMode === 'upload' && !appState.uploadedPdfPath) {
        addMessage('assistant', '请先上传文档');
        return;
    }
    
    // 添加用户消息到聊天
    addMessage('user', message);
    
    // 显示加载动画
    showLoadingAnimation(true);
    updateSystemStatus('processing');
    
    // 清空之前的中间过程消息
    await clearProcessMessages();
    
    // 开始轮询中间过程消息
    startProcessPolling();
    
    // 获取PDF路径
    const pdfPath = appState.currentMode === 'upload' ? 
        appState.uploadedPdfPath : appState.currentPdfPath;
    
    // 发送请求到后端
    try {
        const response = await fetch(API_ENDPOINTS.chat, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                mode: appState.currentMode,
                pdf_path: pdfPath,
                thread_id: appState.threadId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 停止轮询
        stopProcessPolling();
        
        // 隐藏加载动画
        showLoadingAnimation(false);
        updateSystemStatus('ready');
        
        // 添加助手回复（带Markdown渲染）
        addMessage('assistant', data.response);
        
        // 如果有新的PDF路径，更新PDF显示
        if (data.pdf_path) {
            updatePdfDisplay(data.pdf_path);
        }
        
        // 获取最终的过程消息
        await updateProcessMessages();
        
    } catch (error) {
        console.error('Error:', error);
        
        // 停止轮询
        stopProcessPolling();
        
        // 隐藏加载动画
        showLoadingAnimation(false);
        updateSystemStatus('error');
        
        addMessage('assistant', `抱歉，处理您的请求时出错：${error.message}`);
        
        // 添加错误消息到过程显示
        addProcessMessage(`处理请求时出错：${error.message}`, 'error');
    }
    
    updateChatStats();
}

// ========================================
// 添加消息到聊天（支持Markdown渲染）
// ========================================
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    // 根据角色设置头像和名称
    const avatarImage = role === 'user' ? `${API_BASE_URL}/assets/photos/user.jpg` : `${API_BASE_URL}/assets/photos/assistant.jpg`;
    const avatarAlt = role === 'user' ? '用户头像' : '助手头像';
    const senderName = role === 'user' ? '用户' : 'PaperMind';
    
    // 如果是助手消息，渲染Markdown
    let renderedContent = content;
    if (role === 'assistant') {
        // 使用marked.js渲染Markdown
        renderedContent = marked.parse(content);
    } else {
        // 用户消息保持纯文本，但保留换行
        renderedContent = content.replace(/\n/g, '<br>');
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <img src="${avatarImage}" alt="${avatarAlt}" class="avatar-image">
        </div>
        <div class="message-body">
            <div class="message-sender">${senderName}</div>
            <div class="message-content markdown-content">${renderedContent}</div>
        </div>
    `;
    
    elements.chatContent.appendChild(messageDiv);
    
    // 代码高亮
    if (role === 'assistant') {
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
    
    // MathJax渲染数学公式（仅对助手消息）
    if (role === 'assistant' && typeof MathJax !== 'undefined') {
        MathJax.typesetPromise([messageDiv]).catch((err) => {
            console.error('MathJax渲染失败:', err);
        });
    }
    
    // 滚动到底部
    scrollToBottom();
    
    // 更新状态
    appState.messages.push({ role, content });
}

// ========================================
// 添加过程消息
// ========================================
function addProcessMessage(content, level = 'info') {
    const timestamp = new Date().toLocaleTimeString('zh-CN', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `process-message ${level}-message`;
    messageDiv.innerHTML = `
        <div class="process-timestamp">${timestamp}</div>
        <div class="process-text">${content}</div>
    `;
    
    elements.processContent.appendChild(messageDiv);
    
    // 滚动到底部
    elements.processContainer.scrollTop = elements.processContainer.scrollHeight;
}

// ========================================
// 更新过程消息
// ========================================
async function updateProcessMessages() {
    try {
        const response = await fetch(API_ENDPOINTS.getProcessMessages, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.threadId,
                clear: false
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success && data.messages.length > 0) {
            // 清空现有消息
            elements.processContent.innerHTML = '';
            
            // 添加新消息
            data.messages.forEach(msg => {
                addProcessMessage(msg.content, msg.level);
            });
        }
    } catch (error) {
        console.error('获取过程消息失败:', error);
    }
}

// ========================================
// 清空过程消息
// ========================================
async function clearProcessMessages() {
    try {
        const response = await fetch(API_ENDPOINTS.clearProcessMessages, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.threadId
            })
        });
        
        if (response.ok) {
            elements.processContent.innerHTML = '';
            // 添加初始消息
            addProcessMessage('系统已就绪，等待用户输入...', 'info');
        }
    } catch (error) {
        console.error('清空过程消息失败:', error);
    }
}

// ========================================
// 开始轮询过程消息
// ========================================
function startProcessPolling() {
    // 先清空现有消息
    elements.processContent.innerHTML = '';
    addProcessMessage('开始处理用户请求...', 'info');
    
    // 设置轮询间隔
    appState.processPollingInterval = setInterval(async () => {
        await updateProcessMessages();
    }, 1000); // 每秒轮询一次
}

// ========================================
// 停止轮询过程消息
// ========================================
function stopProcessPolling() {
    if (appState.processPollingInterval) {
        clearInterval(appState.processPollingInterval);
        appState.processPollingInterval = null;
    }
}

// ========================================
// 显示/隐藏加载动画
// ========================================
function showLoadingAnimation(show) {
    if (show) {
        elements.loadingAnimation.style.display = 'flex';
        appState.isProcessing = true;
    } else {
        elements.loadingAnimation.style.display = 'none';
        appState.isProcessing = false;
    }
}

// ========================================
// 更新系统状态
// ========================================
function updateSystemStatus(status) {
    elements.statusIndicator.classList.remove('status-ready', 'status-processing', 'status-error');
    
    switch (status) {
        case 'processing':
            elements.statusIndicator.style.background = '#f59e0b';
            elements.statusText.textContent = '处理中...';
            break;
        case 'error':
            elements.statusIndicator.style.background = '#ef4444';
            elements.statusText.textContent = '后端服务未连接';
            break;
        default:
            elements.statusIndicator.style.background = '#4ade80';
            elements.statusText.textContent = '后端服务已连接';
    }
}

// ========================================
// 检查后端连接
// ========================================
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/`, {
            method: 'GET',
            mode: 'cors'
        });
        
        if (response.ok) {
            console.log('后端连接成功');
            updateSystemStatus('ready');
        } else {
            throw new Error('后端返回错误状态');
        }
    } catch (error) {
        console.error('后端连接失败:', error);
        updateSystemStatus('error');
        addProcessMessage('无法连接到后端服务，请确保后端服务已启动', 'error');
    }
}

// ========================================
// 清空聊天
// ========================================
async function clearChat() {
    // 重置当前模式的agent
    try {
        await fetch(API_ENDPOINTS.reset, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: appState.currentMode })
        });
    } catch (error) {
        console.error('重置agent失败:', error);
    }
    
    appState.messages = [{ 
        role: 'assistant', 
        content: '您好，我是您的智能文献助手，可以帮助您检索、分析和解答学术文献相关问题。' 
    }];
    
    elements.chatContent.innerHTML = `
        <div class="message message-assistant">
            <div class="message-avatar">
                <img src="${API_BASE_URL}/assets/photos/assistant.jpg" alt="助手头像" class="avatar-image">
            </div>
            <div class="message-body">
                <div class="message-sender">PaperMind</div>
                <div class="message-content markdown-content">
                    <p>您好，我是您的智能文献助手，可以帮助您检索、分析和解答学术文献相关问题。</p>
                </div>
            </div>
        </div>
    `;
    
    updateChatStats();
}

// ========================================
// 处理模式改变
// ========================================
async function handleModeChange(e) {
    let newMode;
    
    // 判断事件来源
    if (e.target.tagName === 'SELECT') {
        // 来自下拉框
        newMode = e.target.value;
        // 更新隐藏的radio选项以保持同步
        elements.modeOptions.forEach(option => {
            const radio = option.querySelector('input[type="radio"]');
            radio.checked = (option.dataset.mode === newMode);
        });
    } else {
        // 来自radio选项
        const modeOption = e.currentTarget;
        newMode = modeOption.dataset.mode;
        // 更新下拉框以保持同步
        elements.modeSelect.value = newMode;
    }
    
    const oldMode = appState.currentMode;
    
    if (newMode === oldMode) return;
    
    appState.currentMode = newMode;
    
    // 获取模式名称用于提示
    const modeSelect = elements.modeSelect;
    const selectedOption = modeSelect.options[modeSelect.selectedIndex];
    const modeName = selectedOption.text.replace(/^[^\s]+\s*/, '');
    
    // 重置旧模式的agent
    try {
        await fetch(API_ENDPOINTS.reset, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: oldMode })
        });
    } catch (error) {
        console.error('重置agent失败:', error);
    }
    
    // 显示/隐藏上传区域
    if (newMode === 'upload') {
        elements.uploadPanel.style.display = 'flex';
    } else {
        elements.uploadPanel.style.display = 'none';
        // 清除上传的PDF路径
        appState.uploadedPdfPath = null;
    }
    
    // 清空聊天
    clearChat();
    
    // 加载默认PDF
    loadDefaultPdf();
    
    // 清空过程消息
    await clearProcessMessages();
    
    // 添加模式切换提示
    addProcessMessage(`切换到${modeName}模式`, 'info');
}

// ========================================
// 处理PDF上传 - 拖拽事件
// ========================================
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadDropzone.style.filter = 'brightness(1.2)';
    elements.uploadDropzone.style.transform = 'translateY(-2px)';
    elements.uploadDropzone.style.boxShadow = 'var(--shadow-lg)';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadDropzone.style.filter = '';
    elements.uploadDropzone.style.transform = '';
    elements.uploadDropzone.style.boxShadow = '';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    elements.uploadDropzone.style.filter = '';
    elements.uploadDropzone.style.transform = '';
    elements.uploadDropzone.style.boxShadow = '';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handlePdfFile(files[0]);
    }
}

// ========================================
// 处理PDF上传 - 文件选择
// ========================================
async function handlePdfUpload(e) {
    const file = e.target.files[0];
    if (file) {
        await handlePdfFile(file);
    }
}

// ========================================
// 处理PDF文件
// ========================================
async function handlePdfFile(file) {
    if (!file.name.endsWith('.pdf')) {
        showUploadStatus('请选择PDF文件', 'error');
        return;
    }
    
    showUploadStatus('正在上传...', 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        appState.uploadedPdfPath = data.path;
        updatePdfDisplay(data.path, file.name, data.size);
        showUploadStatus(`上传成功：${file.name}`, 'success');
        
        // 添加过程消息
        addProcessMessage(`PDF文件上传成功：${file.name} (${(data.size / 1024).toFixed(1)} KB)`, 'success');
        
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus(`上传失败：${error.message}`, 'error');
        addProcessMessage(`PDF文件上传失败：${error.message}`, 'error');
    }
}

// ========================================
// 显示上传状态
// ========================================
function showUploadStatus(message, type) {
    elements.uploadStatus.textContent = message;
    elements.uploadStatus.className = `upload-status ${type}`;
    elements.uploadStatus.style.display = 'block';
    
    // 5秒后清除状态
    setTimeout(() => {
        elements.uploadStatus.style.display = 'none';
    }, 5000);
}

// ========================================
// 更新PDF显示
// ========================================
function updatePdfDisplay(pdfPath, filename = null, filesize = null) {
    appState.currentPdfPath = pdfPath;
    
    // 清除现有内容
    elements.pdfContainer.innerHTML = '';
    
    // 检查是否是临时文件路径或需要特殊处理的路径
    let pdfUrl;
    if (pdfPath && pdfPath.includes('tmp')) {
        // 如果是临时文件，从后端获取
        const name = pdfPath.split('/').pop();
        pdfUrl = `${API_ENDPOINTS.pdf}/${name}`;
    } else if (pdfPath && pdfPath.includes('awesome_papers')) {
        // 如果是awesome_papers目录下的文件
        const name = pdfPath.split('/').pop();
        pdfUrl = `${API_ENDPOINTS.pdf}/${name}`;
    } else {
        // 显示占位符
        showPdfPlaceholder();
        return;
    }
    
    // 创建PDF预览
    const embed = document.createElement('embed');
    embed.src = pdfUrl;
    embed.width = '100%';
    embed.height = '100%';
    embed.type = 'application/pdf';
    
    elements.pdfContainer.appendChild(embed);
    
    // 更新PDF信息
    if (filename && filesize) {
        const sizeKB = (filesize / 1024).toFixed(1);
        elements.pdfInfo.textContent = `📄 ${filename} (${sizeKB} KB)`;
        elements.pdfInfo.style.display = 'block';
    } else {
        elements.pdfInfo.style.display = 'none';
    }
}

// ========================================
// 加载默认PDF
// ========================================
async function loadDefaultPdf() {
    // 显示占位符
    showPdfPlaceholder();
}

// ========================================
// 显示PDF占位符
// ========================================
function showPdfPlaceholder() {
    elements.pdfContainer.innerHTML = `
        <div class="pdf-placeholder">
            <div class="pdf-placeholder-icon">📄</div>
            <p class="pdf-placeholder-text">选择模式后，这里将显示对应的PDF文档</p>
        </div>
    `;
    elements.pdfInfo.style.display = 'none';
}

// ========================================
// 滚动到底部
// ========================================
function scrollToBottom() {
    const container = elements.chatContainer;
    container.scrollTop = container.scrollHeight;
}

// ========================================
// 更新聊天统计
// ========================================
function updateChatStats() {
    const rounds = Math.floor((appState.messages.length - 1) / 2);
    elements.chatStats.textContent = `对话记录：${rounds} 轮`;
}

// ========================================
// 处理API-KEY提交
// ========================================
async function handleApiKeySubmit() {
    const apiKey = elements.apiKeyInput.value.trim();
    
    if (!apiKey) {
        showApiKeyStatus('请输入API-KEY', 'error');
        return;
    }
    
    // 显示加载状态
    showApiKeyStatus('正在验证API-KEY...', 'info');
    
    try {
        // 发送API-KEY到后端进行验证和设置
        const response = await fetch(`${API_BASE_URL}/set-api-key`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_key: apiKey
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            showApiKeyStatus('API-KEY设置成功', 'success');
            // 清空输入框
            elements.apiKeyInput.value = '';
            // 添加过程消息
            addProcessMessage('API-KEY已更新', 'success');
        } else {
            showApiKeyStatus(`设置失败：${data.message || '未知错误'}`, 'error');
        }
        
    } catch (error) {
        console.error('API-KEY设置错误:', error);
        showApiKeyStatus(`设置失败：${error.message}`, 'error');
    }
}

// ========================================
// 显示API-KEY状态
// ========================================
function showApiKeyStatus(message, type) {
    elements.apiKeyStatus.textContent = message;
    elements.apiKeyStatus.className = `api-key-status ${type}`;
    elements.apiKeyStatus.style.display = 'block';
    
    // 不再自动清除状态，保持持久显示
    // 用户可以手动清除或通过重新设置API-KEY来更新状态
}

// ========================================
// 清除API-KEY状态
// ========================================
function clearApiKeyStatus() {
    elements.apiKeyStatus.style.display = 'none';
    elements.apiKeyStatus.textContent = '';
    elements.apiKeyStatus.className = 'api-key-status';
}

// ========================================
// 公告栏功能
// ========================================

// 设置公告栏事件监听器
function setupAnnouncementEventListeners() {
    // 公告项点击事件
    const announcementItems = elements.announcementContainer.querySelectorAll('.announcement-item');
    announcementItems.forEach(item => {
        item.addEventListener('click', () => {
            const announcementId = item.getAttribute('data-id');
            showAnnouncementModal(announcementId);
        });
    });
    
    // 弹窗关闭按钮事件
    elements.announcementModalClose.addEventListener('click', closeAnnouncementModal);
    
    // 弹窗遮罩层点击事件
    elements.announcementModalOverlay.addEventListener('click', closeAnnouncementModal);
    
    // ESC键关闭弹窗
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.announcementModal.style.display === 'flex') {
            closeAnnouncementModal();
        }
    });
}

// 显示公告弹窗
async function showAnnouncementModal(announcementId) {
    // 根据ID确定文件路径
    let fileName;
    let title;
    
    switch (announcementId) {
        case 'product-intro':
            fileName = '产品简介.md';
            title = '《产品简介》';
            break;
        case 'contact-info':
            fileName = '作者联系方式.md';
            title = '《作者联系方式》';
            break;
        case 'version-update':
            fileName = '版本更新公告.md';
            title = '《版本更新公告》';
            break;
        default:
            console.error('未知的公告ID:', announcementId);
            return;
    }
    
    // 设置标题
    elements.announcementModalTitle.textContent = title;
    
    // 显示加载状态
    elements.announcementModalContent.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-muted);">正在加载公告内容...</div>';
    
    // 显示弹窗
    elements.announcementModal.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // 防止背景滚动
    
    try {
        // 通过后端静态文件服务读取公告文件内容
        const filePath = `${API_BASE_URL}/assets/notes/${fileName}`;
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const content = await response.text();
        
        // 使用marked.js渲染Markdown
        const renderedContent = marked.parse(content);
        
        // 设置内容
        elements.announcementModalContent.innerHTML = renderedContent;
        
        // 代码高亮
        elements.announcementModalContent.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
    } catch (error) {
        console.error('加载公告内容失败:', error);
        elements.announcementModalContent.innerHTML = `
            <div style="text-align: center; padding: 40px; color: var(--text-error);">
                <p>加载公告内容失败</p>
                <p style="font-size: 12px; margin-top: 10px;">错误信息: ${error.message}</p>
            </div>
        `;
    }
}

// 关闭公告弹窗
function closeAnnouncementModal() {
    elements.announcementModal.style.display = 'none';
    document.body.style.overflow = ''; // 恢复背景滚动
}

// ========================================
// 生成会话ID
// ========================================
function generateSessionId() {
    // 使用浏览器指纹生成唯一会话ID
    // 组合用户代理、语言、时区、屏幕分辨率等信息
    const userAgent = navigator.userAgent;
    const language = navigator.language;
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const screenRes = `${screen.width}x${screen.height}`;
    
    // 创建指纹字符串
    const fingerprint = `${userAgent}|${language}|${timezone}|${screenRes}`;
    
    // 生成简单哈希
    let hash = 0;
    for (let i = 0; i < fingerprint.length; i++) {
        const char = fingerprint.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // 转换为32位整数
    }
    
    // 添加时间戳确保唯一性
    const timestamp = Date.now();
    const sessionId = `user_${Math.abs(hash)}_${timestamp}`;
    
    console.log(`生成会话ID: ${sessionId}`);
    return sessionId;
}

// ========================================
// 页面加载完成后初始化
// ========================================
document.addEventListener('DOMContentLoaded', initApp);
