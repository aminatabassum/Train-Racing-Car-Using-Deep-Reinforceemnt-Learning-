$(document).ready(() => {
  $(".tile").click((e) => {
    const id = $(e.currentTarget).attr("id");
    let [r, c] = id.split("-");
    c = parseInt(c);
    r = parseInt(r);

    let unchecked_data = {
      c: c,
      r: r,
      val: $("#tile-type").val(),
    };

    validate_and_post("/update-grid", {}, ajax_post_redirect, unchecked_data);
  });

  $("#reset").click((e) => {
    validate_and_post("/reset-grid", {}, ajax_post_redirect);
  });

  $("#save").click((e) => {
    const data = {
      "save-path": ["empty"],
    };

    validate_and_post("/save-grid", data, swal_ajax_post);
  });

  $("#back").click((e) => {
    validate_and_post("/back", {}, ajax_post_redirect);
  });
});
