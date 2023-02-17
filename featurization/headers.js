const fs = require('fs');

function makeFeature(frames, object, aspect, coord) {
  const f = [];
  for (let mes of ['min', 'max', 'mean', 'med', 'std']) {
    f.push(object + aspect + coord + mes);
  }
  return f;
}

function makeFeatures(pre) {
  var features = [];
  for (let object of ['h', 'l', 'r']) {
    for (let aspect of ['p', 'r']) {
      for (let coord of ['x', 'y', 'z']) {
        features = features.concat(makeFeature(null, object, aspect, coord));
      }
    }
    features = features.concat(makeFeature(null, object, 'r', 'w'));
  }
  return features.map(f => pre + f);
}

function coords(feature) {
  return [feature + 'x', feature + 'y', feature + 'z']
}

function makeNoteFeatures() {
  var feature = ['nid', 'cutDirection', 'colorType', 'noteLineLayer', 'lineIndex', 'scoringType'];
  feature = feature.concat(['saberType', 'saberSpeed', 'timeDeviation', 'cutDirDeviation', 'cutDistanceToCenter', 'cutAngle', 'beforeCutRating', 'afterCutRating'])
  feature = feature.concat(coords('saberDir'))
  feature = feature.concat(coords('cutPoint'))
  feature = feature.concat(coords('cutNormal'))
  return feature;
}

fs.writeFileSync('headers.csv', ['uid'].concat(makeNoteFeatures()).concat(makeFeatures('b')).concat(makeFeatures('a')).join(',') + "\n");
