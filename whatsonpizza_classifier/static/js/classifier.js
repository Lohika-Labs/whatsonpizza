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
                                  '<input type="hidden" name="image" value="' + value + '">' +
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

$( document ).ready(function() {
    console.log("ready");
    $('.class-image').mouseover( function(e){
        $('.images_example img')[0].src = e.target.src;
    });
    $('.class-image').click( function(e){
        $(e.target).closest(".form-check").find(".form-check-input")[0].click();
    })
});

function Boot() {
    classifier = new ClassifierAPI();
    classifier.get_render_page();
    console.log("Boot completed");
}
