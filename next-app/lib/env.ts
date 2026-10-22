export const env = {
  VPN_MANAGER_URL: process.env.VPN_MANAGER_URL ?? "",
  VPN_MANAGER_TOKEN: process.env.VPN_MANAGER_TOKEN ?? "",
};

export function assertEnv() {
  if (!env.VPN_MANAGER_URL || !env.VPN_MANAGER_TOKEN) {
    throw new Error("Faltan variables de entorno: VPN_MANAGER_URL / VPN_MANAGER_TOKEN");
  }
}
