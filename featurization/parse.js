const fs = require('fs');
const process = require('process');
const bsor = require('./open-replay-decoder.js');
const features = require('./features.js');
const utils = require('./utils.js');

function getSamples(user, replays) {
  const samples = [];
  for (const replay of replays) {
    if (replay.includes('bsor')) {
      const file = fs.readFileSync("Z:/beatleader/replays/" + replay);
      try {
        const data = bsor.decode(file.buffer);
        if (data && data.frames && data.frames.length && data.notes && data.notes.length && data.info) {
          const endTime = data.frames[data.frames.length - 1].time;
          for (const note of data.notes) {
            if (note.noteCutInfo && note.noteCutInfo.speedOK && note.noteCutInfo.directionOK && note.noteCutInfo.saberTypeOK && note.eventTime > 1 && note.eventTime < (endTime - 1)) {
              let sample = [user];
              sample = sample.concat(features.makeNoteFeatures(note));
              const framesBefore = utils.fastTimeSlice(data.frames, note.eventTime - 1, note.eventTime);
              sample = sample.concat(features.makeMotionFeatures(framesBefore));
              const framesAfter = utils.fastTimeSlice(data.frames, note.eventTime, note.eventTime + 1);
              sample = sample.concat(features.makeMotionFeatures(framesAfter));
              if (!sample.includes(null) && !sample.includes(undefined) && !sample.includes(NaN)) {
                samples.push(sample);
              }
            }
          }
        }
      } catch (err) { console.error(err) }
    }
  }
  return samples;
}

const id = parseInt(process.argv[2]);
const sessions = JSON.parse(fs.readFileSync('../data/sessions.json', 'utf8'));
const users = Object.keys(sessions);

function handleUser(user, set, count) {
  const samples = getSamples(user, sessions[user][set]);
  utils.shuffle(samples);
  const selected = samples.slice(0, count);
  fs.writeFileSync('../data/' + set + '/' + user + '.csv', selected.map(r => r.join(",")).join("\n") + "\n");
}

const t0 = performance.now();
for (let i = 0; i < users.length; i++) {
  if (i % 32 == id) {
    const user = users[i];
    handleUser(user, 'train', 150);
    handleUser(user, 'validate', 5);
    handleUser(user, 'cluster', 50);
    handleUser(user, 'test', 50);
    console.log(i / users.length);
  }
}
const t1 = performance.now();
const time = t1 - t0;
console.log('Featurization finished in time: ', time);
fs.writeFileSync('../stats/featurization/' + id + '.txt', time.toString());
