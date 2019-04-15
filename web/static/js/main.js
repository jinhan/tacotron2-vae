var sw;
var wavesurfer;

var defaultSpeed = 0.03;
var defaultAmplitude = 0.3;

var activeColors = [[32,133,252], [94,252,169], [253,71,103]];
var inactiveColors = [[241,243,245], [206,212,218], [222,226,230], [173,181,189]];

function generate(ip, port, text, n, s, h, a, condition_on_ref, ref_audio) {//}, speaker_id) {
  $("#synthesize").addClass("is-loading");

  var uri = 'http://' + ip + ':' + port
  var url = uri + '/generate?text=' + encodeURIComponent(text) + "&n=" + n + "&s=" + s + "&h=" + h + "&a=" + a + "&con=" + condition_on_ref + "&ref=" + ref_audio;
  console.log(url);
  fetch(url, {cache: 'no-cache', mode: 'cors'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      var url = URL.createObjectURL(blob);
      inProgress = true;
      console.log(url);
      wavesurfer.load(url);
      $("#synthesize").removeClass("is-loading");
    }).catch(function(err) {
      console.log(err);
      // console.log("error error");
      inProgress = false;
      $("#synthesize").removeClass("is-loading");
    });
}

(function(window, document, undefined){
  window.onload = init;


  function init(){

    wavesurfer = WaveSurfer.create({
      container: '#waveform',
      waveColor: '#017AFD',
      barWidth: 5,
      progressColor: 'navy',
      cursorColr: '#fff',
      normalize:true,
    });

    wavesurfer.on('ready', function () {

      wavesurfer.play();
    });

    wavesurfer.on('finish', function () {
      
    });

    wavesurfer.on('audioprocess', function () {
      if(wavesurfer.isPlaying()) {
        var totalTime = wavesurfer.getDuration(),
          currentTime = wavesurfer.getCurrentTime();
                  
        var timer_total = document.getElementById('time-total');
        var mins = Math.floor(totalTime / 60);
        var secs = Math.floor(totalTime % 60);
        if (secs < 10) {
          secs = '0' + String(secs);
        }
        timer_total.innerText = mins + ':' + secs;

        var timer_current = document.getElementById('time-current');
          var mins = Math.floor(currentTime / 60);
          var secs = Math.floor(currentTime % 60);
          if (secs < 10) {
            secs = '0' + String(secs);
          }
          timer_current.innerText = mins + ':' + secs;
      }
    });

    var loadFile = {
      contents:"Null",
      init: function() {
        $.ajax({
          url:"/uploads/koemo_spk_emo_all_test.txt",
          dataType: "text",
          async:false, 
          success: function(data) {
            var allText = data;
            var split = allText.split('\n')
            var randomNum = Math.floor(Math.random() * split.length);
            var randomLine = split[randomNum];
            loadFile.contents = randomLine.split('|')[0].replace('/data1/jinhan', '/uploads');
          }
        });
      }
    }
    
    var condition_on_ref = false;

    $(document).ready(function() {
      

      $("#mix").click(function() {
        $(".SliderFrame").css("display", "");
        $(".RefAudioFrame").css("display", "none");
        condition_on_ref = false;
      });
      $("#refaudio").click(function() {
        $(".SliderFrame").css("display", "none");
        $(".RefAudioFrame").css("display", "");
        condition_on_ref = true;
      });

      $("#neu").change(function(){
        var neu = this.value;
      });
      $("#sad").change(function(){
        var sad = this.value;
      });
      $("#hap").change(function(){
        var hap = this.value;
      });
      $("#ang").change(function(){
        var ang = this.value;
      });

      $('#random-audio').click(function() {
        loadFile.init();
        console.log(loadFile.contents);

        var audio = document.getElementById('audio');
        audio.src = loadFile.contents;
        console.log(audio.src);
        audio.load();
      });
    });


    $(document).on('click', "#synthesize", function() {
      synthesize();
    });

    function synthesize() {
      var text = $("#text").val().trim();
      var text_length = text.length;
      var ref_audio = $("#audio").attr('src');

      generate('10.100.1.119', 51000, text, neu.value, sad.value, hap.value, ang.value, condition_on_ref, ref_audio);

      var lowpass = wavesurfer.backend.ac.createGain();
      wavesurfer.backend.setFilter(lowpass);
    }

  }
})(window, document, undefined);
