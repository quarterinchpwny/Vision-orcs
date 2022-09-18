!function(r){r.languages.xquery=r.languages.extend("markup",{"xquery-comment":{pattern:/\(:[\s\S]*?:\)/,greedy:!0,alias:"comment"},string:{pattern:/(["'])(?:\1\1|(?!\1)[\s\S])*\1/,greedy:!0},extension:{pattern:/\(#.+?#\)/,alias:"symbol"},variable:/\$[-\w:]+/,axis:{pattern:/(^|[^-])(?:ancestor(?:-or-self)?|attribute|child|descendant(?:-or-self)?|following(?:-sibling)?|parent|preceding(?:-sibling)?|self)(?=::)/,lookbehind:!0,alias:"operator"},"keyword-operator":{pattern:/(^|[^:-])\b(?:and|castable as|div|eq|except|ge|gt|idiv|instance of|intersect|is|le|lt|mod|ne|or|union)\b(?=$|[^:-])/,lookbehind:!0,alias:"operator"},keyword:{pattern:/(^|[^:-])\b(?:as|ascending|at|base-uri|boundary-space|case|cast as|collation|construction|copy-namespaces|declare|default|descending|else|empty (?:greatest|least)|encoding|every|external|for|function|if|import|in|inherit|lax|let|map|module|namespace|no-inherit|no-preserve|option|order(?: by|ed|ing)?|preserve|return|satisfies|schema|some|stable|strict|strip|then|to|treat as|typeswitch|unordered|validate|variable|version|where|xquery)\b(?=$|[^:-])/,lookbehind:!0},function:/[\w-]+(?::[\w-]+)*(?=\s*\()/,"xquery-element":{pattern:/(element\s+)[\w-]+(?::[\w-]+)*/,lookbehind:!0,alias:"tag"},"xquery-attribute":{pattern:/(attribute\s+)[\w-]+(?::[\w-]+)*/,lookbehind:!0,alias:"attr-name"},builtin:{pattern:/(^|[^:-])\b(?:attribute|comment|document|element|processing-instruction|text|xs:(?:anyAtomicType|anyType|anyURI|base64Binary|boolean|byte|date|dateTime|dayTimeDuration|decimal|double|duration|ENTITIES|ENTITY|float|gDay|gMonth|gMonthDay|gYear|gYearMonth|hexBinary|ID|IDREFS?|int|integer|language|long|Name|NCName|negativeInteger|NMTOKENS?|nonNegativeInteger|nonPositiveInteger|normalizedString|NOTATION|positiveInteger|QName|short|string|time|token|unsigned(?:Byte|Int|Long|Short)|untyped(?:Atomic)?|yearMonthDuration))\b(?=$|[^:-])/,lookbehind:!0},number:/\b\d+(?:\.\d+)?(?:E[+-]?\d+)?/,operator:[/[+*=?|@]|\.\.?|:=|!=|<[=<]?|>[=>]?/,{pattern:/(\s)-(?=\s)/,lookbehind:!0}],punctuation:/[[\](){},;:/]/}),r.languages.xquery.tag.pattern=/<\/?(?!\d)[^\s>\/=$<%]+(?:\s+[^\s>\/=]+(?:=(?:("|')(?:\\[\s\S]|\{(?!\{)(?:\{(?:\{[^{}]*\}|[^{}])*\}|[^{}])+\}|(?!\1)[^\\])*\1|[^\s'">=]+))?)*\s*\/?>/i,r.languages.xquery.tag.inside["attr-value"].pattern=/=(?:("|')(?:\\[\s\S]|\{(?!\{)(?:\{(?:\{[^{}]*\}|[^{}])*\}|[^{}])+\}|(?!\1)[^\\])*\1|[^\s'">=]+)/i,r.languages.xquery.tag.inside["attr-value"].inside.punctuation=/^="|"$/,r.languages.xquery.tag.inside["attr-value"].inside.expression={pattern:/\{(?!\{)(?:\{(?:\{[^{}]*\}|[^{}])*\}|[^{}])+\}/,inside:r.languages.xquery,alias:"language-xquery"};var s=function(e){return"string"==typeof e?e:"string"==typeof e.content?e.content:e.content.map(s).join("")},l=function(e){for(var t=[],n=0;n<e.length;n++){var a=e[n],o=!1;if("string"!=typeof a&&("tag"===a.type&&a.content[0]&&"tag"===a.content[0].type?"</"===a.content[0].content[0].content?0<t.length&&t[t.length-1].tagName===s(a.content[0].content[1])&&t.pop():"/>"===a.content[a.content.length-1].content||t.push({tagName:s(a.content[0].content[1]),openedBraces:0}):!(0<t.length&&"punctuation"===a.type&&"{"===a.content)||e[n+1]&&"punctuation"===e[n+1].type&&"{"===e[n+1].content||e[n-1]&&"plain-text"===e[n-1].type&&"{"===e[n-1].content?0<t.length&&0<t[t.length-1].openedBraces&&"punctuation"===a.type&&"}"===a.content?t[t.length-1].openedBraces--:"comment"!==a.type&&(o=!0):t[t.length-1].openedBraces++),(o||"string"==typeof a)&&0<t.length&&0===t[t.length-1].openedBraces){var i=s(a);n<e.length-1&&("string"==typeof e[n+1]||"plain-text"===e[n+1].type)&&(i+=s(e[n+1]),e.splice(n+1,1)),0<n&&("string"==typeof e[n-1]||"plain-text"===e[n-1].type)&&(i=s(e[n-1])+i,e.splice(n-1,1),n--),/^\s+$/.test(i)?e[n]=i:e[n]=new r.Token("plain-text",i,null,i)}a.content&&"string"!=typeof a.content&&l(a.content)}};r.hooks.add("after-tokenize",function(e){"xquery"===e.language&&l(e.tokens)})}(Prism);