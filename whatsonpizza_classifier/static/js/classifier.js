function ClassifierAPI() {

    this.render_page = function(result) {
        var images;
        html = [];
        images = result['data'];
        welcome_msg = 'Welcome, ' + result['username'] + ': ' + result.progress.processed  + '/' + result.progress.queue_max
        $('title').html(welcome_msg);
        $('div.status').html(welcome_msg);
        $.each(images, function(idx, value) {
            if (idx == 0) {
                html.push('<div class="row">');
            } else if (idx % 4 == 0 ) {
                html.push('</div><div class="row">');
            }
            html.push('<div class="col-sm-6 col-md-3"><span>' +
                            '<div class="thumbnail img">' +
                                  '<img src="/image/' + value +'">' +
                                  '<div class="switch_btn btn </div>' +
                                  '<input type="hidden" name="' + value + '" value="' + value + '">' +
                             '</div>' +
                      '</span></div>');
        });
        html.push('</div>');
        $("div.images").html(html.join('\n'));
    }

    this.get_render_page = function() {
        var render_page = this.render_page;
        request = $.getJSON('/images');
        request.done(function(data) {
           render_page(data);
        });
    };
}


function Boot() {
    classifier = new ClassifierAPI();
    classifier.get_render_page();
    console.log("Boot completed");
}
