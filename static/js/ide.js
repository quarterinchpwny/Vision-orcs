let editor;

window.onload = function() {
    editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/java");
}

function executeCode() {

    $.ajax({
        url: "/predict/run",

        method: "POST",

        data: {
            code: editor.getSession().getValue()
        },

        success: function(response) {
            $(".output").text(response)
        }

    })
}