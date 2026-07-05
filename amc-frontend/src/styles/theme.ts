import { createTheme } from "@mui/material/styles";
import type { ThemeOptions } from "@mui/material/styles";

const getDesignTokens = (mode: "light" | "dark"): ThemeOptions => ({
  palette: {
    mode,
    ...(mode === "light"
      ? {
          primary: {
            main: "#0052ff", // Electric Indigo Blue
            light: "#4785ff",
            dark: "#003eb8",
          },
          secondary: {
            main: "#0e7490", // Cyan Accent
            light: "#06b6d4",
            dark: "#155e75",
          },
          background: {
            default: "#f8fafc", // Clean slate default
            paper: "rgba(255, 255, 255, 0.8)", // Glassmorphic translucent paper
          },
          text: {
            primary: "#0f172a",
            secondary: "#475569",
          },
        }
      : {
          primary: {
            main: "#00d2ff", // Vivid Neon Cyber Cyan
            light: "#67e8f9",
            dark: "#0891b2",
          },
          secondary: {
            main: "#a855f7", // Cyber Purple Accent
            light: "#c084fc",
            dark: "#7e22ce",
          },
          background: {
            default: "#070a13", // Deep Space Cybersecurity Dark background
            paper: "rgba(13, 20, 38, 0.55)", // Translucent Cyber Blue/Black glass
          },
          text: {
            primary: "#f8fafc",
            secondary: "#94a3b8",
          },
        }),
  },
  typography: {
    fontFamily: '"Outfit", "Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 800,
      fontSize: "2.5rem",
      letterSpacing: "-0.03em",
    },
    h2: {
      fontWeight: 700,
      fontSize: "2rem",
      letterSpacing: "-0.02em",
    },
    h5: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
    h6: {
      fontWeight: 600,
      letterSpacing: "-0.01em",
    },
    button: {
      textTransform: "none",
      fontWeight: 600,
      letterSpacing: "0.02em",
    },
  },
  shape: {
    borderRadius: 16, // Softer rounded corners for modern tech card look
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          padding: "8px 18px",
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            transform: "translateY(-1.5px)",
            boxShadow:
              mode === "light"
                ? "0 6px 20px rgba(0, 82, 255, 0.2)"
                : "0 6px 20px rgba(0, 210, 255, 0.3)",
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          border:
            mode === "light"
              ? "1px solid rgba(226, 232, 240, 0.8)"
              : "1px solid rgba(255, 255, 255, 0.07)",
          boxShadow:
            mode === "light"
              ? "0 8px 32px 0 rgba(148, 163, 184, 0.08)"
              : "0 8px 32px 0 rgba(0, 0, 0, 0.35)",
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            transform: "translateY(-3px)",
            boxShadow:
              mode === "light"
                ? "0 12px 40px 0 rgba(148, 163, 184, 0.16)"
                : "0 12px 40px 0 rgba(0, 210, 255, 0.15)", // Electric cyber glowing border effect
            borderColor:
              mode === "light"
                ? "rgba(0, 82, 255, 0.3)"
                : "rgba(0, 210, 255, 0.25)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        outlined: {
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          background:
            mode === "light"
              ? "rgba(255, 255, 255, 0.7)"
              : "rgba(13, 20, 38, 0.45)",
          border:
            mode === "light"
              ? "1px solid rgba(226, 232, 240, 0.8)"
              : "1px solid rgba(255, 255, 255, 0.07)",
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-notchedOutline": {
            borderColor:
              mode === "light"
                ? "rgba(148, 163, 184, 0.35)"
                : "rgba(255, 255, 255, 0.16)",
            transition: "border-color 0.2s ease, box-shadow 0.2s ease",
          },
          "&:hover .MuiOutlinedInput-notchedOutline": {
            borderColor:
              mode === "light"
                ? "rgba(0, 82, 255, 0.35)"
                : "rgba(0, 210, 255, 0.35)",
          },
          "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
            borderColor:
              mode === "light"
                ? "rgba(0, 82, 255, 0.7)"
                : "rgba(0, 210, 255, 0.7)",
            boxShadow:
              mode === "light"
                ? "0 0 0 3px rgba(0, 82, 255, 0.12)"
                : "0 0 0 3px rgba(0, 210, 255, 0.16)",
          },
        },
        icon: {
          color:
            mode === "light"
              ? "rgba(15, 23, 42, 0.75)"
              : "rgba(248, 250, 252, 0.75)",
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
          background:
            mode === "light"
              ? "rgba(255, 255, 255, 0.82)"
              : "rgba(7, 10, 19, 0.78)",
          border:
            mode === "light"
              ? "1px solid rgba(226, 232, 240, 0.8)"
              : "1px solid rgba(255, 255, 255, 0.08)",
          boxShadow:
            mode === "light"
              ? "0 12px 32px rgba(15, 23, 42, 0.12)"
              : "0 12px 32px rgba(0, 0, 0, 0.35)",
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          color: mode === "light" ? "#0f172a" : "#f8fafc",
          "&:hover": {
            backgroundColor:
              mode === "light"
                ? "rgba(0, 82, 255, 0.08)"
                : "rgba(0, 210, 255, 0.12)",
          },
          "&.Mui-selected": {
            backgroundColor:
              mode === "light"
                ? "rgba(0, 82, 255, 0.14)"
                : "rgba(0, 210, 255, 0.16)",
            "&:hover": {
              backgroundColor:
                mode === "light"
                  ? "rgba(0, 82, 255, 0.2)"
                  : "rgba(0, 210, 255, 0.2)",
            },
          },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundImage: "none",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          background:
            mode === "light"
              ? "rgba(255, 255, 255, 0.85)"
              : "rgba(7, 10, 19, 0.8)",
          borderRight:
            mode === "light"
              ? "1px solid rgba(226, 232, 240, 0.8)"
              : "1px solid rgba(255, 255, 255, 0.06)",
        },
      },
    },
  },
});

export const getTheme = (mode: "light" | "dark") =>
  createTheme(getDesignTokens(mode));
export default getTheme;
