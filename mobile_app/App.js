import { StatusBar } from "expo-status-bar";
import * as DocumentPicker from "expo-document-picker";
import { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

export default function App() {
  const [apiUrl, setApiUrl] = useState("http://192.168.1.2:8000");
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function chooseVideo() {
    const selection = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
    });
    if (!selection.canceled) {
      setVideo(selection.assets[0]);
      setResult(null);
    }
  }

  async function analyzeVideo() {
    if (!video) {
      Alert.alert("Video select karein", "Prediction ke liye pehle video choose karein.");
      return;
    }

    setLoading(true);
    setResult(null);
    try {
      const data = new FormData();
      data.append("file", {
        uri: video.uri,
        name: video.name || "video.mp4",
        type: video.mimeType || "video/mp4",
      });

      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        body: data,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }
      setResult(payload);
    } catch (error) {
      Alert.alert("Prediction error", error.message);
    } finally {
      setLoading(false);
    }
  }

  const isReview = result?.prediction === "review";
  const accent = isReview ? "#d97706" : result?.is_deepfake ? "#dc2626" : "#16a34a";

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <View style={styles.container}>
        <Text style={styles.eyebrow}>VIDEO AUTHENTICITY CHECK</Text>
        <Text style={styles.title}>Deepfake Detector</Text>
        <Text style={styles.subtitle}>
          Video upload karein aur trained model se authenticity score check karein.
        </Text>

        <View style={styles.card}>
          <Text style={styles.label}>API address</Text>
          <TextInput
            autoCapitalize="none"
            autoCorrect={false}
            onChangeText={setApiUrl}
            style={styles.input}
            value={apiUrl}
          />
          <Text style={styles.hint}>Laptop ka local IP use karein, localhost nahi.</Text>

          <TouchableOpacity onPress={chooseVideo} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Choose video</Text>
          </TouchableOpacity>
          <Text numberOfLines={2} style={styles.filename}>
            {video ? video.name : "Abhi koi video selected nahi hai"}
          </Text>

          <TouchableOpacity
            disabled={loading}
            onPress={analyzeVideo}
            style={[styles.primaryButton, loading && styles.disabled]}
          >
            {loading ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.primaryButtonText}>Analyze video</Text>
            )}
          </TouchableOpacity>
        </View>

        {result && (
          <View style={[styles.resultCard, { borderColor: accent }]}>
            <Text style={styles.resultLabel}>MODEL RESULT</Text>
            <Text style={[styles.resultTitle, { color: accent }]}>
              {isReview
                ? "Needs manual review"
                : result.is_deepfake
                  ? "Deepfake detected"
                  : "Likely real video"}
            </Text>
            <Text style={styles.score}>
              {isReview
                ? "Model is not confident enough"
                : `${(result.confidence * 100).toFixed(1)}% confidence`}
            </Text>
            <Text style={styles.detail}>
              Fake score: {(result.fake_probability * 100).toFixed(1)}% | Faces found:{" "}
              {result.valid_frames}/{result.sampled_frames}
            </Text>
          </View>
        )}

        <Text style={styles.footer}>
          Result automated estimate hai. Sensitive decisions ke liye manual verification bhi karein.
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { backgroundColor: "#f8fafc", flex: 1 },
  container: { flex: 1, padding: 24, paddingTop: 56 },
  eyebrow: { color: "#2563eb", fontSize: 12, fontWeight: "800", letterSpacing: 1.6 },
  title: { color: "#0f172a", fontSize: 34, fontWeight: "800", marginTop: 8 },
  subtitle: { color: "#475569", fontSize: 16, lineHeight: 24, marginTop: 8 },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 20,
    elevation: 3,
    marginTop: 28,
    padding: 18,
    shadowColor: "#0f172a",
    shadowOpacity: 0.08,
    shadowRadius: 14,
  },
  label: { color: "#334155", fontSize: 13, fontWeight: "700" },
  input: {
    backgroundColor: "#f1f5f9",
    borderRadius: 10,
    color: "#0f172a",
    fontSize: 15,
    marginTop: 8,
    padding: 12,
  },
  hint: { color: "#64748b", fontSize: 12, marginTop: 7 },
  secondaryButton: {
    alignItems: "center",
    borderColor: "#2563eb",
    borderRadius: 12,
    borderWidth: 1,
    marginTop: 22,
    padding: 13,
  },
  secondaryButtonText: { color: "#2563eb", fontSize: 15, fontWeight: "700" },
  filename: { color: "#475569", fontSize: 13, marginTop: 10, textAlign: "center" },
  primaryButton: {
    alignItems: "center",
    backgroundColor: "#2563eb",
    borderRadius: 12,
    marginTop: 18,
    padding: 15,
  },
  primaryButtonText: { color: "#ffffff", fontSize: 16, fontWeight: "800" },
  disabled: { opacity: 0.65 },
  resultCard: {
    backgroundColor: "#ffffff",
    borderRadius: 18,
    borderWidth: 2,
    marginTop: 22,
    padding: 18,
  },
  resultLabel: { color: "#64748b", fontSize: 11, fontWeight: "800", letterSpacing: 1.2 },
  resultTitle: { fontSize: 24, fontWeight: "800", marginTop: 7 },
  score: { color: "#0f172a", fontSize: 18, fontWeight: "700", marginTop: 8 },
  detail: { color: "#64748b", fontSize: 13, lineHeight: 20, marginTop: 8 },
  footer: { color: "#64748b", fontSize: 12, lineHeight: 18, marginTop: "auto" },
});
