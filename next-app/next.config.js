/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // App Router es el default; dejamos el flag por claridad.
    serverActions: { allowedOrigins: ["*"] },
  },
};

module.exports = nextConfig;
