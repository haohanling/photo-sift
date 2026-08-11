"""生成交互式 HTML 验证报告。

特点（解决人工复核的痛点）：
- 每张照片一张卡片：缩略图 + 判定 + 特征数值 + 理由
- 点击缩略图 → 页内放大显示**原图**（无需切到原文件夹）
- 顶部筛选按钮：只看 保留 / 需人工复核 / 剔除
- 每张卡片有【保留】【删除】按钮：点击后**真正移动文件**并更新判定。
  该功能需要配合 review_server.py（`python -m photosift.cli <dir> --serve`）使用；
  直接 file:// 打开报告时按钮不可用（不加载服务器）。

报告可通过两种方式打开：
  1. file:// 直接打开（只读查看）
  2. http://127.0.0.1:<port>（服务器提供，支持按钮 + 原图 + 状态持久化）
"""

import base64
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .features import PhotoFeatures

THUMB_MAX_EDGE = 360

CATEGORY_STYLE = {
    "keep": ("保留", "#e8f5e9", "#2e7d32"),
    "reject_blur": ("剔除·模糊", "#fdecea", "#c62828"),
    "reject_closed_eyes": ("剔除·闭眼", "#fdecea", "#c62828"),
    "reject_exposure": ("剔除·曝光", "#fdecea", "#c62828"),
    "reject_manual": ("人工删除", "#eceff1", "#546e7a"),
    "llm": ("待复核", "#fff8e1", "#f9a825"),
    "": ("待复核", "#fff8e1", "#f9a825"),
}

# 非保留（需要人工复核）的类别
REVIEW_CATS = ("reject_blur", "reject_closed_eyes", "reject_exposure", "reject_manual", "llm", "")

# 传给 JS 的标签/颜色表
_JS_LABELS = {k: {"label": v[0], "bg": v[1], "fg": v[2]} for k, v in CATEGORY_STYLE.items()}


def _encode_thumb(path: Path, max_edge: int = THUMB_MAX_EDGE) -> str:
    d = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(d, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(1.0, max_edge / max(h, w))
    img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return base64.b64encode(buf.tobytes()).decode()


def _orig_src(photo_dir: Path, filename: str, out_path: Path) -> str:
    """生成 file:// 打开时指向原图的相对路径。"""
    rel = os.path.relpath(str(photo_dir), str(out_path.parent)).replace("\\", "/")
    return f"{rel}/{html.escape(filename)}"


def generate(photos: list[PhotoFeatures], photo_dir: Path, out_path: Path) -> Path:
    """生成报告，返回文件路径。"""
    cards = []
    counts = {}
    for p in photos:
        cat = p.category or p.verdict
        counts[cat] = counts.get(cat, 0) + 1
        lab, bg, fg = CATEGORY_STYLE.get(cat, (cat, "#eee", "#333"))
        thumb = _encode_thumb(p.path)
        orig = _orig_src(photo_dir, p.filename, out_path)
        raw_name = p.filename
        name = html.escape(p.filename)
        reason = html.escape(p.reason or "")
        b, e, y = p.blur, p.exposure, p.eyes
        ear = ""
        if y.get("eyes_ear"):
            ear = " EAR=" + ",".join(
                str(t) for pair in y["eyes_ear"] for t in pair
            )
        cards.append(
            f"""<div class="card" data-cat="{cat}" data-fname="{html.escape(raw_name, quote=True)}"
     data-orig="{orig}">
      <img class="thumb" src="data:image/jpeg;base64,{thumb}" alt="{name}" onclick="openLightbox(this)"/>
      <div class="meta">
        <div class="fname">{name}</div>
        <span class="badge" style="background:{bg};color:{fg}">{lab}</span>
        <div class="feat">模糊 {b.get('verdict','-')} {b.get('laplacian_variance','-')}</div>
        <div class="feat">曝光 {e.get('verdict','-')} mean={e.get('mean_brightness','-')}</div>
        <div class="feat">人脸 {y.get('faces',0)} 睁眼={y.get('all_eyes_open','-')}{ear}</div>
        <div class="reason">{reason}</div>
        <div class="votebar">
          <button class="vbtn keep" onclick="event.stopPropagation();vote(this,'keep')">👍 保留</button>
          <button class="vbtn del"  onclick="event.stopPropagation();vote(this,'reject')">🗑 删除</button>
        </div>
      </div>
    </div>"""
        )

    total = len(photos)
    n_review = sum(counts.get(c, 0) for c in REVIEW_CATS)
    n_keep = counts.get("keep", 0)
    js_labels = json.dumps(_JS_LABELS, ensure_ascii=False)

    html_doc = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>photo-sift 验证报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, "Microsoft YaHei", sans-serif; background:#f5f5f5; margin:0; padding:20px; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .sub {{ color:#666; font-size:13px; margin-bottom:12px; }}
  .chips {{ display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }}
  .chip {{ padding:6px 14px; border-radius:16px; border:1px solid #ccc; background:#fff; cursor:pointer; font-size:13px; }}
  .chip.active {{ background:#1976d2; color:#fff; border-color:#1976d2; }}
  .wrap {{ display:flex; flex-wrap:wrap; }}
  .card {{ width:200px; border:1px solid #ddd; border-radius:8px; overflow:hidden; margin:8px;
           background:#fff; cursor:zoom-in; transition:box-shadow .15s; }}
  .card:hover {{ box-shadow:0 2px 8px rgba(0,0,0,.15); }}
  .thumb {{ width:100%; display:block; }}
  .meta {{ padding:8px; font-size:12px; line-height:1.55; }}
  .fname {{ font-weight:700; font-size:11px; }}
  .badge {{ display:inline-block; margin:4px 0; padding:1px 8px; border-radius:4px; font-weight:600; }}
  .feat {{ color:#555; }}
  .reason {{ color:#888; margin-top:2px; }}
  .votebar {{ display:flex; gap:6px; margin-top:6px; }}
  .vbtn {{ flex:1; padding:4px 0; border:none; border-radius:5px; cursor:pointer; font-size:12px; font-weight:600; }}
  .vbtn.keep {{ background:#e8f5e9; color:#2e7d32; }}
  .vbtn.del  {{ background:#fdecea; color:#c62828; }}
  .vbtn:hover {{ filter:brightness(.94); }}
  .vbtn:disabled {{ opacity:.5; cursor:default; }}
  #lightbox {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:100;
                flex-direction:column; align-items:center; justify-content:center; padding:20px; }}
  #lightbox img {{ max-width:94vw; max-height:82vh; object-fit:contain; }}
  #lb-bar {{ color:#fff; margin-top:10px; font-size:14px; text-align:center; }}
  #lb-bar a {{ color:#90caf9; margin-left:10px; }}
  .lb-close {{ position:absolute; top:16px; right:24px; color:#fff; font-size:34px; cursor:pointer; line-height:1; }}
  .toast {{ position:fixed; bottom:24px; left:50%; transform:translateX(-50%); background:#323232; color:#fff;
            padding:10px 18px; border-radius:6px; font-size:13px; display:none; z-index:200; }}
</style></head><body>
<h1>photo-sift 验证报告</h1>
<div class="sub">共 {total} 张 · 保留 {n_keep} · 需人工复核 {n_review} · 点击缩略图放大原图 · 用👍保留/🗑删除调整判定</div>
<div class="chips">
  <button class="chip active" onclick="filter('all')">全部 ({total})</button>
  <button class="chip" onclick="filter('keep')">保留 ({n_keep})</button>
  <button class="chip" onclick="filter('review')">需人工复核 ({n_review})</button>
  <button class="chip" onclick="filter('reject_blur')">剔除·模糊</button>
  <button class="chip" onclick="filter('reject_closed_eyes')">剔除·闭眼</button>
  <button class="chip" onclick="filter('reject_exposure')">剔除·曝光</button>
</div>
<div class="wrap">{"".join(cards)}</div>

<div id="lightbox" onclick="if(event.target===this)closeLightbox()">
  <span class="lb-close" onclick="closeLightbox()">&times;</span>
  <img id="lb-img" alt=""/>
  <div id="lb-bar"><span id="lb-name"></span><span id="lb-badge"></span><span id="lb-reason"></span>
    <a id="lb-open" href="#" target="_blank">在新标签打开原图</a></div>
</div>
<div class="toast" id="toast"></div>

<script>
  var LABELS = {js_labels};
  var curFilter = 'all';
  var IS_SERVER = location.protocol.indexOf('http') === 0;

  function origSrc(card){{
    if (IS_SERVER) return '/photo/' + encodeURIComponent(card.dataset.fname);
    return card.dataset.orig;  // file:// 下用相对路径
  }}

  function openLightbox(img){{
    var card = img.closest('.card');
    var name = card.dataset.fname;
    var badge = card.querySelector('.badge').textContent;
    var reason = card.querySelector('.reason').textContent;
    document.getElementById('lb-img').src = origSrc(card);
    document.getElementById('lb-name').textContent = name;
    document.getElementById('lb-badge').textContent = ' [' + badge + '] ';
    document.getElementById('lb-reason').textContent = reason;
    document.getElementById('lb-open').href = origSrc(card);
    document.getElementById('lightbox').style.display = 'flex';
  }}
  function closeLightbox(){{ document.getElementById('lightbox').style.display = 'none'; }}
  document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeLightbox(); }});

  function showToast(msg){{
    var t = document.getElementById('toast');
    t.textContent = msg; t.style.display = 'block';
    clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', 1600);
  }}

  function setCat(card, cat){{
    card.dataset.cat = cat;
    var st = LABELS[cat] || {{label:cat,bg:'#eee',fg:'#333'}};
    var badge = card.querySelector('.badge');
    badge.textContent = st.label;
    badge.style.background = st.bg; badge.style.color = st.fg;
    applyFilter(curFilter);
    recount();
  }}

  function applyFilter(k){{
    curFilter = k;
    document.querySelectorAll('.card').forEach(c => {{
      var cat = c.dataset.cat;
      var show;
      if (k === 'all') show = true;
      else if (k === 'review') show = cat !== 'keep';
      else show = (cat === k);
      c.style.display = show ? '' : 'none';
    }});
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.chip').forEach(c => {{
      if (c.textContent.indexOf(k) !== -1 || (k==='all' && c.textContent.indexOf('全部')!==-1)) c.classList.add('active');
    }});
  }}
  function filter(k){{ applyFilter(k); }}

  function recount(){{
    var byCat = {{}};
    document.querySelectorAll('.card').forEach(c => {{
      var cat = c.dataset.cat;
      byCat[cat] = (byCat[cat]||0) + 1;
    }});
    var total = document.querySelectorAll('.card').length;
    var keep = byCat['keep']||0;
    var review = total - keep;
    document.querySelectorAll('.chip').forEach(ch => {{
      var t = ch.textContent;
      if (t.indexOf('全部')!==-1) ch.textContent = '全部 (' + total + ')';
      else if (t.indexOf('保留')!==-1 && t.indexOf('剔除')===-1) ch.textContent = '保留 (' + keep + ')';
      else if (t.indexOf('需人工复核')!==-1) ch.textContent = '需人工复核 (' + review + ')';
    }});
  }}

  function vote(btn, action){{
    if (!IS_SERVER){{
      showToast('请用服务器模式打开才能调整判定：python -m photosift.cli <照片目录> --serve');
      return;
    }}
    var card = btn.closest('.card');
    var name = card.dataset.fname;
    btn.disabled = true;
    fetch('/api/vote', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{filename:name, action:action}})
    }}).then(r => r.json()).then(data => {{
      btn.disabled = false;
      if (data.ok){{ setCat(card, data.category); showToast((action==='keep'?'已保留：':'已删除：') + name); }}
      else showToast('失败：' + (data.error||'未知错误'));
    }}).catch(() => {{ btn.disabled = false; showToast('请求失败，服务器是否还在运行？'); }});
  }}

  // 服务器模式下，加载时从 verdicts.json 同步最新判定（刷新后保持）
  if (IS_SERVER){{
    fetch('/api/state').then(r => r.json()).then(state => {{
      document.querySelectorAll('.card').forEach(c => {{
        var cat = state[c.dataset.fname];
        if (cat) setCat(c, cat);
      }});
    }}).catch(()=>{{}});
  }}
</script>
</body></html>"""

    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
