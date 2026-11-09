// Browser AudioWorklet: 16 kHz mono PCM16, exactly 512 samples per message.
class PCM16Encoder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.phase = 0;
    this.sum = 0;
    this.count = 0;
    this.frame = new Int16Array(512);
    this.pos = 0;
    this.step = 16000 / sampleRate;
  }
  process(inputs, outputs) {
    for (const output of outputs) for (const channel of output) channel.fill(0);
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i];
      this.count++;
      this.phase += this.step;
      if (this.phase < 1) continue;
      this.phase -= 1;
      const averaged = this.sum / this.count;
      this.sum = 0;
      this.count = 0;
      const value = Math.max(-1, Math.min(1, averaged));
      this.frame[this.pos++] = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
      if (this.pos === 512) {
        this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
        this.frame = new Int16Array(512);
        this.pos = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm16-encoder", PCM16Encoder);
