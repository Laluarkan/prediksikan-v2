const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("show");
    });
  }

  const profileToggle = document.getElementById("profile-toggle");
  const profileMenu = document.getElementById("profile-menu");
  if (profileToggle && profileMenu) {
    profileToggle.addEventListener("click", () => {
      profileMenu.classList.toggle("show");
    });
    window.addEventListener("click", (e) => {
      if (!profileToggle.contains(e.target) && !profileMenu.contains(e.target)) {
        profileMenu.classList.remove("show");
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const flashMessages = document.querySelectorAll('.flash-msg');
    flashMessages.forEach(msg => {
      setTimeout(() => {
          msg.classList.add('fade-out');
          setTimeout(() => {
              msg.remove();
          }, 500); 
      }, 2000); // 2 detik
    });
  });