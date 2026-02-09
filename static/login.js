// ✅ Fill these from your Supabase project settings
const SUPABASE_URL = "{{ supabase_url }}";
const SUPABASE_ANON_KEY = "{{ supabase_anon_key }}";

const supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function login() {
    document.getElementById("err").textContent = "";

    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value;

    const email = username + "@gmail.com";

    const { data, error } = await supabaseClient.auth.signInWithPassword({ email, password });

    if (error) {
        document.getElementById("err").textContent = error.message;
        return;
    }

    const token = data.session.access_token;

    // Create Flask session
    const resp = await fetch("/session", {
        method: "POST",
        headers: { "Authorization": "Bearer " + token }
    });

    if (!resp.ok) {
        document.getElementById("err").textContent = "Login failed (server rejected token).";
        return;
    }

    window.location.href = "/";
}