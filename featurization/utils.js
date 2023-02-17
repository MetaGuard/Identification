function lastIndexBelow(frames, start) {
  if (frames[0].time >= start) return -1;
  let low = 0; high = frames.length;
  while (low != high) {
    const mid = Math.ceil((low + high) / 2);
    if (frames[mid].time >= start) high = mid - 1;
    else low = mid;
  }
  return low;
}

function firstIndexAbove(frames, end) {
  if (frames[frames.length - 1].time <= end) return frames.length;
  let low = 0; high = frames.length;
  while (low != high) {
    const mid = Math.floor((low + high) / 2);
    if (frames[mid].time <= end) low = mid + 1;
    else high = mid;
  }
  return high;
}

function fastTimeSlice(frames, start, end) {
  return frames.slice(lastIndexBelow(frames, start) + 1, firstIndexAbove(frames, end));
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function derive(frames) {
  for (let i = 0; i < frames.length; i++) {
    for (let object of ['h', 'l', 'r']) {
      for (let aspect of ['p', 'e', 'r']) {
        for (let coord of ['x', 'y', 'z']) {
          const p = frames[i][object][aspect][coord];
          let v = 0;
          if (frames.length == 1) {
          } else if (i == 0) {
            const x1 = frames[0][object][aspect][coord];
            const x2 = frames[1][object][aspect][coord];
            v = x2 - x1;
          } else if (i == frames.length - 1) {
            const x1 = frames[frames.length - 2][object][aspect][coord];
            const x2 = frames[frames.length - 1][object][aspect][coord];
            v = x2 - x1;
          } else {
            const x1 = frames[i - 1][object][aspect][coord];
            const x2 = frames[i + 1][object][aspect][coord];
            v = (x2 - x1) / 2;
          }
          frames[i][object][aspect][coord] = {p, v};
        }
      }
    }
  }
}

module.exports.fastTimeSlice = fastTimeSlice;
module.exports.shuffle = shuffle;
module.exports.derive = derive;
