function makeMotionFeature(frames, object, aspect, coord) {
  const values = frames.map(frame => frame[object][aspect][coord]);
  values.sort();
  const min = values[0];
  const max = values[values.length - 1];

  const half = Math.floor(values.length / 2);
  const med = (values.length % 2) ? values[half] : ((values[half - 1] + values[half]) / 2.0);

  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  const mean = sum / values.length;

  let squares = 0;
  for (let i = 0; i < values.length; i++) {
    squares += ((values[i] - mean) ** 2);
  }
  const std = Math.sqrt(squares / values.length);

  return [min, max, mean, med, std];
}

function makeMotionFeatures(frames) {
  var features = [];
  for (let object of ['h', 'l', 'r']) {
    for (let aspect of ['p', 'r']) {
      for (let coord of ['x', 'y', 'z']) {
        features = features.concat(makeMotionFeature(frames, object, aspect, coord));
      }
    }
    features = features.concat(makeMotionFeature(frames, object, 'r', 'w'));
  }

  return features;
}

function coords(feature) {
  return [feature.x, feature.y, feature.z];
}

function makeNoteFeatures(note) {
  let note_id = note.noteID
  let x = note_id
  let cutDirection = (x % 10)
  x = (x - cutDirection) / 10
  let colorType = (x % 10)
  x = (x - colorType) / 10
  let noteLineLayer = (x % 10)
  x = (x - noteLineLayer) / 10
  let lineIndex = (x % 10)
  x = (x - lineIndex) / 10
  let scoringType = (x % 10)
  var feature = ['N' + note_id, cutDirection, colorType, noteLineLayer, lineIndex, scoringType];
  let cut = note.noteCutInfo
  feature = feature.concat([cut.saberType, cut.saberSpeed, cut.timeDeviation, cut.cutDirDeviation, cut.cutDistanceToCenter, cut.cutAngle, cut.beforeCutRating, cut.afterCutRating])
  feature = feature.concat(coords(cut.saberDir))
  feature = feature.concat(coords(cut.cutPoint))
  feature = feature.concat(coords(cut.cutNormal))
  return feature;
}

module.exports.makeMotionFeatures = makeMotionFeatures;
module.exports.makeNoteFeatures = makeNoteFeatures;
