$(document).ready(function () {
  $("#generate").click(function () {
    const data = {
      nr: ["empty"],
      nc: ["empty"],
    };

    validate_and_post("/generate", data, swal_ajax_post_redirect);
  });

  $("#edit").click(function () {
    const data = {
      path: ["empty"],
    };

    validate_and_post("/edit", data, swal_ajax_post_redirect);
  });
});
