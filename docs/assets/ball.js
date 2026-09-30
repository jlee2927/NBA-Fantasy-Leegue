/* Scroll-driven basketball for the NBA Fantasy Leegue landing page.
 *
 * The ball is drawn from scratch rather than projected from a photograph, so
 * there is no product image to license and nothing to download: the whole
 * thing is about 4kB of maths instead of a 1.4MB JPEG.
 *
 * How it works. Every pixel inside the circle is treated as a point on a
 * sphere, so a pixel at (nx, ny) has depth nz = sqrt(1 - nx^2 - ny^2). Those
 * coordinates never change, so they are worked out once. Each frame the point
 * is spun around the vertical axis by the scroll angle, and the rotated
 * position decides whether it lands on a seam and how brightly it is lit.
 */
(function () {
  'use strict';

  var stage = document.querySelector('[data-ball-stage]');
  var canvas = document.querySelector('[data-ball]');
  if (!stage || !canvas) return;

  var context = canvas.getContext('2d');
  if (!context) return;

  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  // Leather, lit and unlit, and the near-black of a seam.
  var LEATHER = [206, 98, 43];
  var SEAM = [38, 24, 18];
  var LIGHT = [-0.34, 0.44, 0.83];   // pointing at the ball from upper left, front
  var TURNS = 3;                      // full rotations across the scroll stage

  // Seam geometry. The two straight seams are great circles through the poles;
  // the two curved ones ride up and down in latitude, bulging where they meet
  // the straight pair, which is what gives a basketball its shape.
  var SEAM_WIDTH = 0.030;
  var CURVE_CENTRE = 0.46;
  var CURVE_SWING = 0.27;

  var pixels, nx, ny, nz, alpha, grain, size = 0;
  var frame = 0, lastAngle = null;

  function prepare() {
    var wanted = Math.min(720, Math.round(
      canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 1.5)));
    if (!wanted || wanted === size) return false;

    size = wanted;
    canvas.width = size;
    canvas.height = size;
    pixels = context.createImageData(size, size);

    var count = size * size;
    nx = new Float32Array(count);
    ny = new Float32Array(count);
    nz = new Float32Array(count);
    alpha = new Float32Array(count);
    grain = new Float32Array(count);

    // One pixel of softness at the silhouette, so the edge is not jagged.
    var feather = 2 / size;

    for (var y = 0; y < size; y++) {
      var py = (2 * (y + 0.5) / size) - 1;
      for (var x = 0; x < size; x++) {
        var px = (2 * (x + 0.5) / size) - 1;
        var squared = px * px + py * py;
        if (squared >= 1) continue;

        var i = y * size + x;
        var radius = Math.sqrt(squared);
        nx[i] = px;
        ny[i] = py;
        nz[i] = Math.sqrt(1 - squared);
        alpha[i] = Math.min(1, (1 - radius) / feather);
        // Fixed speckle standing in for the pebbled surface. It does not spin
        // with the ball, but at this strength it reads as texture, not as a
        // pattern, and it costs nothing per frame.
        grain[i] = (Math.random() - 0.5) * 0.05;
      }
    }
    lastAngle = null;
    return true;
  }

  function draw(angle) {
    if (!pixels) return;
    var data = pixels.data;
    var cos = Math.cos(angle);
    var sin = Math.sin(angle);

    for (var i = 0; i < alpha.length; i++) {
      var out = i << 2;
      var a = alpha[i];
      if (!a) { data[out + 3] = 0; continue; }

      var x = nx[i], y = ny[i], z = nz[i];

      // Spin around the vertical axis. y is untouched, so a point's latitude
      // never changes - only which way it faces.
      var rx = x * cos - z * sin;
      var rz = x * sin + z * cos;

      // Straight seams: the two great circles running pole to pole.
      var seam = Math.abs(rx) < SEAM_WIDTH || Math.abs(rz) < SEAM_WIDTH;

      // Curved seams. The horizontal radius is 1 - y^2, and cos(2 * longitude)
      // works out as (rx^2 - rz^2) over that, which avoids calling atan2 on
      // every pixel of every frame.
      if (!seam) {
        var flat = rx * rx + rz * rz;
        if (flat > 1e-6) {
          var swing = CURVE_CENTRE - CURVE_SWING * ((rx * rx - rz * rz) / flat);
          seam = Math.abs(Math.abs(y) - swing) < SEAM_WIDTH;
        }
      }

      // Lambert shading, lifted off black by an ambient term, plus a little
      // extra light where the surface faces us most directly.
      var lit = x * LIGHT[0] + y * LIGHT[1] + z * LIGHT[2];
      if (lit < 0) lit = 0;
      var shade = 0.30 + 0.78 * lit * lit + grain[i];

      var colour = seam ? SEAM : LEATHER;
      var r = colour[0] * shade;
      var g = colour[1] * shade;
      var b = colour[2] * shade;

      data[out] = r > 255 ? 255 : r;
      data[out + 1] = g > 255 ? 255 : g;
      data[out + 2] = b > 255 ? 255 : b;
      data[out + 3] = a * 255;
    }
    context.putImageData(pixels, 0, 0);
  }

  function update() {
    frame = 0;
    var resized = prepare();

    var angle = 0;
    if (!reducedMotion.matches) {
      var box = stage.getBoundingClientRect();
      var travel = Math.max(1, box.height - window.innerHeight);
      var progress = Math.min(1, Math.max(0, -box.top / travel));
      angle = progress * Math.PI * 2 * TURNS;
    }

    // Skip frames the eye could not tell apart.
    if (!resized && lastAngle !== null && Math.abs(angle - lastAngle) < 0.002) return;
    lastAngle = angle;
    draw(angle);
  }

  function onScroll() {
    if (!frame) frame = requestAnimationFrame(update);
  }

  update();
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  reducedMotion.addEventListener('change', onScroll);
})();
