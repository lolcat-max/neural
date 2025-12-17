// arduino_analog_id_streamer_shift.ino
// Analog -> integer token ID streamer (bitshift-based).
//
// PC -> Arduino (ASCII, newline-terminated):
//   PING
//   V<integer>   e.g. V5000
//   N<integer>   e.g. N300000
//   GO
//
// Arduino -> PC:
//   READY (repeated until configured)
//   PONG
//   OK V=...
//   OK N=...
//   START
//   <id>   (N lines)
//   DONE

const int ANALOG_PIN = A0;

// Typical AVR ADC is 10-bit => 0..1023 [web:135][web:145]
const uint16_t ADC_MIN = 0;
const uint16_t ADC_MAX = 1023;
const uint8_t  ADC_BITS = 10;

long V = -1; // vocab size
long N = -1; // how many ids to output

String readLine() {
  if (!Serial.available()) return String();
  String s = Serial.readStringUntil('\n');
  s.trim();
  return s;
}

static inline bool isPowerOfTwoU32(uint32_t x) {
  return x && ((x & (x - 1)) == 0);
}

static inline uint8_t log2_u32_pow2(uint32_t x) {
  // x must be power-of-two
  uint8_t k = 0;
  while (x >>= 1) k++;
  return k;
}

uint32_t analogToTokenShift(uint16_t adcValue, uint32_t vocabSize) {
  // Clamp ADC into expected range (defensive).
  if (adcValue < ADC_MIN) adcValue = ADC_MIN;
  if (adcValue > ADC_MAX) adcValue = ADC_MAX;

  if (vocabSize <= 1) return 0;

  // Fast path: vocab is power-of-two => just take top bits (bitshift quantizer).
  // Example: V=256 => keep top 8 bits of the 10-bit ADC: id = adc >> 2.
  if (isPowerOfTwoU32(vocabSize)) {
    uint8_t k = log2_u32_pow2(vocabSize); // vocabSize == 2^k

    if (k == 0) return 0;

    if (k >= ADC_BITS) {
      // If V > 1024 (k > 10), upscale ADC into more bits by left shifting
      // then mask into range [0, V-1].
      uint8_t lshift = k - ADC_BITS;
      uint32_t id = ((uint32_t)adcValue) << lshift;
      return id & (vocabSize - 1);
    } else {
      uint8_t rshift = ADC_BITS - k;
      return ((uint32_t)adcValue) >> rshift;
    }
  }

  // General path: scale by multiply + right shift (fixed-point), then clamp.
  // id ≈ floor(adc * V / 1024), using a >> 10 shift (fast divide by 1024). [web:151]
  uint32_t id = ((uint32_t)adcValue * (uint32_t)vocabSize) >> ADC_BITS; // >>10
  if (id >= vocabSize) id = vocabSize - 1;
  return id;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(2000);

  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 2000) { }

  pinMode(ANALOG_PIN, INPUT);
}

void loop() {
  static unsigned long lastReady = 0;
  if ((V <= 0 || N <= 0) && millis() - lastReady > 500) {
    Serial.println("READY");
    lastReady = millis();
  }

  String line = readLine();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  if (line.charAt(0) == 'V') {
    V = line.substring(1).toInt();
    Serial.print("OK V=");
    Serial.println(V);
    return;
  }

  if (line.charAt(0) == 'N') {
    N = line.substring(1).toInt();
    Serial.print("OK N=");
    Serial.println(N);
    return;
  }

  if (line == "GO") {
    if (V <= 0 || N <= 0) {
      Serial.println("ERR not configured");
      return;
    }

    Serial.println("START");

    for (long i = 0; i < N; i++) {
      uint16_t adc = (uint16_t)analogRead(ANALOG_PIN);
      uint32_t id  = analogToTokenShift(adc, (uint32_t)V);
      Serial.println(id);

      // If your PC can’t keep up, uncomment:
      // delayMicroseconds(100);
    }

    Serial.println("DONE");
    while (true) { delay(1000); }
  }

  Serial.println("ERR unknown cmd");
}
