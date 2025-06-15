gsap.registerPlugin(ScrollTrigger);

// --- SETUP START ---
// Using Locomotive Scroll from Locomotive https://github.com/locomotivemtl/locomotive-scroll
const locoScroll = new LocomotiveScroll({
el: document.querySelector(".smooth-scroll"),
    smooth: true,
    multiplier: 1
});
// each time Locomotive Scroll updates, tell ScrollTrigger to update too (sync positioning)
locoScroll.on("scroll", ScrollTrigger.update);

// tell ScrollTrigger to use these proxy methods for the ".smooth-scroll" element since Locomotive Scroll is hijacking things
ScrollTrigger.scrollerProxy(".smooth-scroll", {
scrollTop(value) {
    return arguments.length ? locoScroll.scrollTo(value, {duration: 0, disableLerp: true}) : locoScroll.scroll.instance.scroll.y;
}, // we don't have to define a scrollLeft because we're only scrolling vertically.
getBoundingClientRect() {
    return {top: 0, left: 0, width: window.innerWidth, height: window.innerHeight};
},
// LocomotiveScroll handles things completely differently on mobile devices - it doesn't even transform the container at all! So to get the correct behavior and avoid jitters, we should pin things with position: fixed on mobile. We sense it by checking to see if there's a transform applied to the container (the LocomotiveScroll-controlled element).
pinType: document.querySelector(".smooth-scroll").style.transform ? "transform" : "fixed"
});

// each time the window updates, we should refresh ScrollTrigger and then update LocomotiveScroll. 
ScrollTrigger.addEventListener("refresh", () => locoScroll.update());
ScrollTrigger.defaults({ scroller: ".smooth-scroll" });
// --- SETUP END ---
  

// scroll trigger start

ScrollTrigger.matchMedia({
    // desktop
    "(min-width: 1367px)": function () {
        // first section
        var tl = gsap.timeline({scrollTrigger:{
            trigger: ".second-section",
            start: "20% 100%",
            end: "50% 50%",
            scrub: true,
            // markers: true
        }})

        tl.to("#cookie", {
            top: "138%",
            left: "0%",
            rotate: "50%"
        },'cookie')

        tl.to("#chips", {
            width: "8vw",
            top: "125%",
            left: "88%"
        },'cookie')

        // second seciton
        var t2 = gsap.timeline({scrollTrigger:{
            trigger: ".third-section",
            start: "20% 100%",
            end: "50% 50%",
            scrub: true,
            // markers: true
        }})

        t2.to('#cookie',{
            top: "237%",
            left: "43%",
            width: "13vw",
            rotate: "-50deg"
        },"cookism")

        t2.from('#cookism',{
            rotate: "0"
        },"cookism")
        t2.to('#cookism',{
            rotate: "-50deg"
        },"cookism")
    }
})
  
// scroll trigger start

// scroll magic - move to next section on mousewheel

// var ctrl = new ScrollMagic.Controller({   
// });

// This each sets the animation
// $('.slide').each(function(index,element) {  
  
//       ////////// scroll up //////////

//       new ScrollMagic.Scene({
//           triggerHook: 0,
//           triggerElement: this,
//           offset:-50 // small offset added to prevent overscrolling accidentally triggering
//       })
//       .on('leave', function () {
//             TweenLite.to(window, .5, {scrollTo:{y:$(window).height() * (index-1), autoKill:false},ease: Linear.easeNone})
//       })
//       .addTo(ctrl);  // scene end

//       //////////scroll down //////////

//        new ScrollMagic.Scene({
//             triggerElement: this,
//             triggerHook: 0,
//             offset:50 // small offset added to prevent overscrolling accidentally triggering
//         })
//         .on('enter', function () {
//              TweenLite.to(window, .5, {scrollTo:{y:$(window).height() * (index+1), autoKill:false},ease: Linear.easeNone});   
//         })
//         .addTo(ctrl); // scene end

// }); 

//slide each

// scroll magic end