document.querySelectorAll(".assignment-three figure a[href]").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
});

document.querySelectorAll('.assignment-three a[href^="http://"], .assignment-three a[href^="https://"]').forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
});