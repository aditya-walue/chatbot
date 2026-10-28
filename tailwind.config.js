import frappeUIPreset from "frappe-ui/tailwind";

/** @type {import('tailwindcss').Config} */
export default {
	presets: [frappeUIPreset],
	// Desk's bootstrap defines the same utility names (.py-3, .mb-2, …) with
	// !important; flow's must be !important too so the #flow-root prefix can win.
	important: true,
	// Tailwind v3 ignores `content` declared inside presets, so the frappe-ui
	// source globs must be listed here for its component classes to be emitted.
	content: [
		"./frontend/src/**/*.{vue,js}",
		"./node_modules/frappe-ui/src/components/**/*.{vue,js,ts}",
		"./node_modules/frappe-ui/src/composables/**/*.{vue,js,ts}",
		"./node_modules/frappe-ui/src/utils/**/*.{vue,js,ts}",
	],
	theme: {
		extend: {
			fontSize: {
				"2xs": [
					"12px",
					{ lineHeight: "1.15", letterSpacing: "0" },
				],
				xs: ["13px", { lineHeight: "1.5", letterSpacing: "0" }],
				sm: ["14px", { lineHeight: "1.5", letterSpacing: "0" }],
				base: ["15px", { lineHeight: "1.6", letterSpacing: "0" }],
				lg: ["17px", { lineHeight: "1.15", letterSpacing: "0" }],
				xl: ["19px", { lineHeight: "1.15", letterSpacing: "0" }],
				"2xl": [
					"21px",
					{ lineHeight: "1.15", letterSpacing: "0" },
				],
				"3xl": [
					"25px",
					{ lineHeight: "1.15", letterSpacing: "0" },
				],
			},
		},
	},
	plugins: [],
};
