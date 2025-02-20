document.addEventListener("DOMContentLoaded", function() {
    var currentIndex = 0;
    var images = document.querySelectorAll(".image");
    var totalImages = images.length;

    function showImage(index) {
        images.forEach((img, i) => {
            img.style.display = i === index ? "block" : "none";
        });
    }

    document.getElementById("next").addEventListener("click", function() {
        currentIndex = (currentIndex + 1) % totalImages;
        showImage(currentIndex);
    });

    document.getElementById("prev").addEventListener("click", function() {
        currentIndex = (currentIndex - 1 + totalImages) % totalImages;
        showImage(currentIndex);
    });

    showImage(currentIndex);
});