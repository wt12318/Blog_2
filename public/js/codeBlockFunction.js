function load_language() {
    var blocks = $('code.hljs');
    for (var i = 0; i < blocks.length; i++) {
        var span = document.createElement('span');
        span.classList.add('codeblock-language');
        span.innerHTML = blocks[i].classList[1];
        blocks[i].insertBefore(span, blocks[i].children[0]);
    }
}

$(document).ready(load_language);