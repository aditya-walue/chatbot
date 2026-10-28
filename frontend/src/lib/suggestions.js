import { __ } from "@/lib/translate";

// DocTypes that are the panel's or the desk's own, not something to ask about.
const IGNORED = /^(Flow |Workspace|Desktop|Notification Log|Error Log|User$|DocType$)/;
// Standard business DocTypes to fall back on when the user has opened too few lately; only
// the ones this user can read are used, so a card never points at something they can't see.
const COMMON = ["Sales Invoice", "Customer", "Item", "Sales Order", "Purchase Order", "Employee", "Lead", "Project"];

// Cards for the home screen, built from the DocTypes this user opened lately (desk keeps
// them in `frappe.boot.user.recent`), topped up from COMMON. `text` is what gets sent;
// `title` and `hint` are what the card shows.
export function suggestedQuestions(limit = 4) {
	const [first, second = first] = relevantDocTypes();
	const cards = [];
	if (first) {
		cards.push({
			icon: "file-text",
			tone: "green",
			title: __("Latest {0}", [first]),
			hint: __("Most recent records"),
			text: __("Show the latest {0} records", [first]),
		});
	}
	cards.push({
		icon: "users",
		tone: "blue",
		title: __("Who has which role?"),
		hint: __("Users and permissions"),
		text: __("What roles do I have?"),
	});
	if (second) {
		cards.push({
			icon: "box",
			tone: "amber",
			title: __("How many {0}?", [second]),
			hint: __("Count the records"),
			text: __("How many {0} records are there?", [second]),
		});
		// A how-to for the sales invoice when this user can read it, else for the second DocType.
		const guide = canRead("Sales Invoice") ? "Sales Invoice" : second;
		cards.push({
			icon: "plus-circle",
			tone: "violet",
			title: __("Create a new {0}", [guide]),
			hint: __("Step by step"),
			text: __("How do I create a new {0}?", [guide]),
		});
	}
	return cards.slice(0, limit);
}

function relevantDocTypes() {
	const types = [];
	for (const doctype of [...recentDocTypes(), ...COMMON.filter(canRead)]) {
		if (!types.includes(doctype)) types.push(doctype);
		if (types.length === 2) break;
	}
	return types;
}

function canRead(doctype) {
	const readable = frappe?.boot?.user?.can_read;
	return Array.isArray(readable) && readable.includes(doctype);
}

function recentDocTypes() {
	let recent = frappe?.boot?.user?.recent;
	try {
		if (typeof recent === "string") recent = JSON.parse(recent);
	} catch {
		return [];
	}
	if (!Array.isArray(recent)) return [];
	const types = [];
	for (const entry of recent) {
		const doctype = Array.isArray(entry) ? entry[0] : null;
		if (doctype && !IGNORED.test(doctype) && !types.includes(doctype)) types.push(doctype);
		if (types.length === 2) break;
	}
	return types;
}
